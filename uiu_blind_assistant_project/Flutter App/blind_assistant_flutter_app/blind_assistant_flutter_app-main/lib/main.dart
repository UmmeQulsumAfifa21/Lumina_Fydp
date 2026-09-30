import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:translator/translator.dart';
import 'package:url_launcher/url_launcher.dart';

// The gap starts AFTER the complete feedback finishes, including both languages.
// All speech, including SOS, Explore, and manual replay, uses this same gate.
const voiceFeedbackGap = Duration(seconds: 12);
const statusPollInterval = Duration(milliseconds: 750);
const gpsUploadInterval = Duration(seconds: 5);

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const BlindAssistantApp());
}

Duration Function() monotonicClock() {
  final watch = Stopwatch()..start();
  return () => watch.elapsed;
}

class VoiceJob {
  const VoiceJob({
    required this.channel,
    required this.english,
    this.bangla,
    this.priority = 0,
    this.isValid,
  });

  final String channel;
  final String english;
  final String? bangla;
  final int priority;
  final bool Function()? isValid;
  bool get valid => isValid?.call() ?? true;
}

/// One pending item per channel. New guidance replaces old guidance.
/// No urgent bypass: even a new SOS must wait for the shared quiet period.
class FeedbackScheduler extends ChangeNotifier {
  FeedbackScheduler({
    required this.speak,
    Duration Function()? clock,
    this.gap = voiceFeedbackGap,
    this.onError,
  }) : _clock = clock ?? monotonicClock();

  final Future<void> Function(VoiceJob) speak;
  final Duration Function() _clock;
  final Duration gap;
  final void Function(Object)? onError;
  final Map<String, VoiceJob> _pending = {};
  Timer? _timer;
  Duration? _finishedAt;
  bool _disposed = false;
  bool _suspended = false;
  bool busy = false;
  String? activeChannel;

  int get pendingCount => _pending.length;
  Duration get remaining {
    if (_finishedAt == null) return Duration.zero;
    final left = gap - (_clock() - _finishedAt!);
    return left.isNegative ? Duration.zero : left;
  }

  void submit(VoiceJob job) {
    if (_disposed || _suspended || job.english.trim().isEmpty) return;
    _pending[job.channel] = job;
    _pump();
  }

  void remove(String channel) => _pending.remove(channel);

  void clear() {
    _pending.clear();
    _timer?.cancel();
    _timer = null;
  }

  void suspend(bool value) {
    _suspended = value;
    if (value) clear();
    if (!value) _pump();
    if (!_disposed) notifyListeners();
  }

  void _pump() {
    if (_disposed || _suspended || busy) return;
    _timer?.cancel();
    _timer = null;
    _pending.removeWhere((_, job) => !job.valid);
    if (_pending.isEmpty) return;
    if (remaining > Duration.zero) {
      _timer = Timer(remaining, _pump);
      return;
    }
    final next = _pending.values.reduce(
          (a, b) => a.priority >= b.priority ? a : b,
    );
    _pending.remove(next.channel);
    busy = true;
    activeChannel = next.channel;
    notifyListeners();
    unawaited(_run(next));
  }

  Future<void> _run(VoiceJob job) async {
    try {
      await speak(job);
    } catch (error) {
      onError?.call(error);
    } finally {
      // Also cool down after a cancelled/failed attempt: never create a retry
      // loop that talks continuously when a plugin or translation fails.
      _finishedAt = _clock();
      busy = false;
      activeChannel = null;
      if (!_disposed) {
        notifyListeners();
        _pump();
      }
    }
  }

  @override
  void dispose() {
    _disposed = true;
    clear();
    super.dispose();
  }
}

Map<String, dynamic> objectMap(dynamic value) =>
    value is Map ? Map<String, dynamic>.from(value) : <String, dynamic>{};

double? finiteNumber(dynamic value) {
  final result = value is num ? value.toDouble() : double.tryParse('$value');
  return result != null && result.isFinite ? result : null;
}

String? normalizeServerAddress(String input) {
  var value = input.trim();
  if (value.isEmpty || RegExp(r'\s').hasMatch(value)) return null;
  if (!value.contains('://')) value = 'http://$value';
  final uri = Uri.tryParse(value);
  if (uri == null ||
      uri.host.isEmpty ||
      uri.userInfo.isNotEmpty ||
      !['http', 'https'].contains(uri.scheme) ||
      uri.hasQuery ||
      uri.hasFragment ||
      (uri.path.isNotEmpty && uri.path != '/')) {
    return null;
  }
  final port = uri.hasPort ? uri.port : 5000;
  if (port < 1 || port > 65535) return null;
  return Uri(scheme: uri.scheme, host: uri.host, port: port).toString();
}

class AppSettings {
  const AppSettings({
    required this.server,
    required this.language,
    required this.emergencyNumber,
    required this.handsFree,
  });
  final String server;
  final String language;
  final String emergencyNumber;
  final bool handsFree;
}

class AssistantController extends ChangeNotifier {
  AssistantController({
    http.Client? client,
    this.enableDeviceServices = true,
    Future<void> Function(VoiceJob)? voiceOutput,
    this.smsOutput,
    Duration Function()? clock,
  }) : _client = client ?? http.Client(),
        _clock = clock ?? monotonicClock() {
    feedback = FeedbackScheduler(
      clock: _clock,
      speak: voiceOutput ?? _speakJob,
      onError: (error) {
        audioStatus = 'Voice unavailable. Check the phone’s speech engine.';
        _notify();
      },
    )..addListener(_onSpeechState);
    feedback.suspend(enableDeviceServices);
  }

  final http.Client _client;
  final bool enableDeviceServices;
  final Future<void> Function(String phone, String body)? smsOutput;
  final Duration Function() _clock;
  final _tts = FlutterTts();
  final _speech = stt.SpeechToText();
  final _translator = GoogleTranslator();
  static const _smsChannel = MethodChannel('blind_assistant/sms');
  late final FeedbackScheduler feedback;
  SharedPreferences? _preferences;
  final _translations = <String, String>{};
  final Set<Completer<void>> _requests = {};
  final Set<String> _handledSos = {};
  Timer? _pollTimer, _gpsTimer, _voiceRestart, _clockTimer;
  StreamSubscription<Position>? _positionSubscription;
  Completer<void>? _stopUtterance;
  bool _disposed = false,
      _initializing = false,
      _pollBusy = false,
      _gpsBusy = false;
  bool _speechAvailable = false, _startingMic = false, _handlingSos = false;
  bool _foreground = true, _ttsReady = false;
  int _generation = 0, _speechGeneration = 0;
  Duration? _lastStatusAt, _exploreStartedAt, _lastSosReminder;
  String? _requestedExploreId;
  String? _handledExploreId;
  String? _lastConnectionNotice;
  String _banglaLocale = 'bn-BD';
  bool _banglaAvailable = false;

  String server = 'http://10.105.129.254:5000';
  String language = 'en';
  String emergencyNumber = '';
  bool handsFree = true, voiceEnabled = true, microphoneListening = false;
  bool serverOnline = false, exploreStarting = false, acknowledging = false;
  String connectionStatus = 'Connecting to your server';
  String guidance = 'Waiting for a fresh view of your surroundings.';
  String gpsStatus = 'Waiting for location permission';
  String audioStatus = 'Preparing voice feedback';
  String recognizedSpeech = '';
  String smsStatus = 'No SOS event received';
  String? lastError;
  String? lastSpoken;
  String? explorationDescription;
  Position? position;
  DateTime? lastGpsUpload;
  Map<String, dynamic> normal = {},
      camera = {},
      performance = {},
      gemini = {},
      explore = {};

  bool get freshConnection =>
      serverOnline &&
          _lastStatusAt != null &&
          _clock() - _lastStatusAt! < const Duration(seconds: 4);
  bool get cameraLive => freshConnection && camera['available'] == true;
  bool get exploring => exploreStarting || normal['mode'] == 'explore';
  bool get sosActive =>
      normal['sos_latched'] == true || normal['sos_pressed'] == true;
  bool get sosPressed => freshConnection && normal['sos_pressed'] == true;
  double? get distance =>
      cameraLive ? finiteNumber(normal['distance_raw_cm']) : null;
  List<Map<String, dynamic>> get detections =>
      cameraLive && normal['status'] == 'ok'
          ? (normal['detections'] is List
          ? (normal['detections'] as List).map(objectMap).toList()
          : [])
          : [];
  String get languageLabel => language == 'bn'
      ? 'বাংলা'
      : language == 'both'
      ? 'English + বাংলা'
      : 'English';
  int get quietSeconds => (feedback.remaining.inMilliseconds / 1000).ceil();
  String get voiceStatus => !voiceEnabled
      ? 'Voice feedback paused'
      : feedback.busy
      ? 'Speaking now'
      : quietSeconds > 0
      ? 'Next voice slot in ${quietSeconds}s'
      : 'Ready for fresh guidance';

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  Future<void> initialize() async {
    if (_initializing || _disposed) return;
    _initializing = true;
    try {
      _preferences = await SharedPreferences.getInstance();
      if (_disposed) return;
      server =
          normalizeServerAddress(
            _preferences!.getString('python_server_base_url') ?? '',
          ) ??
              server;
      final savedLanguage = _preferences!.getString('audio_language');
      language = ['en', 'bn', 'both'].contains(savedLanguage)
          ? savedLanguage!
          : 'en';
      emergencyNumber = _preferences!.getString('sos_phone_number') ?? '';
      handsFree = _preferences!.getBool('hands_free_enabled') ?? true;
      voiceEnabled = _preferences!.getBool('voice_feedback_enabled') ?? true;
      _handledSos.addAll(
        _preferences!.getStringList('handled_sos_events_v2') ?? [],
      );
    } catch (_) {
      lastError = 'Saved settings could not be loaded.';
    }
    if (_disposed) return;
    _notify();
    unawaited(pollStatus());
    _clockTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!freshConnection && serverOnline) {
        _offline('Waiting for fresh server data');
      }
      _notify();
    });
    if (enableDeviceServices) {
      // Camera polling starts before potentially slow OS permission dialogs.
      await _initializeTts();
      if (_disposed) return;
      await _initializeGps();
      if (_disposed) return;
      await _initializeSpeech();
    } else {
      _ttsReady = true;
    }
    if (_disposed) return;
    feedback.suspend(!voiceEnabled || !_ttsReady);
    _gpsTimer = Timer.periodic(
      gpsUploadInterval,
          (_) => unawaited(uploadGps()),
    );
    _scheduleListening();
    _notify();
  }

  Future<void> _initializeTts() async {
    try {
      await _tts.awaitSpeakCompletion(true);
      await _tts.setLanguage('en-US');
      await _tts.setSpeechRate(.48);
      await _tts.setVolume(1);
      await _tts.setPitch(1);
      for (final locale in ['bn-BD', 'bn-IN']) {
        final available = await _tts.isLanguageAvailable(locale);
        if (available == true || available == 1) {
          _banglaLocale = locale;
          _banglaAvailable = true;
          break;
        }
      }
      _ttsReady = true;
      audioStatus = '12 seconds of quiet after every voice message';
      feedback.suspend(!voiceEnabled);
    } catch (_) {
      audioStatus = 'Text-to-speech could not start';
    }
    _notify();
  }

  Future<Map<String, dynamic>> requestJson(
      String path, {
        Map<String, dynamic>? body,
        Duration timeout = const Duration(seconds: 3),
      }) async {
    final endpoint = server;
    final generation = _generation;
    final cancel = Completer<void>();
    _requests.add(cancel);
    final timer = Timer(timeout, () {
      if (!cancel.isCompleted) cancel.complete();
    });
    try {
      final request = http.AbortableRequest(
        body == null ? 'GET' : 'POST',
        Uri.parse('$endpoint$path'),
        abortTrigger: cancel.future,
      );
      request.headers['Accept'] = 'application/json';
      if (body != null) {
        request.headers['Content-Type'] = 'application/json';
        request.body = jsonEncode(body);
      }
      final response = await http.Response.fromStream(
        await _client.send(request),
      );
      if (_disposed || generation != _generation) {
        throw StateError('Request no longer current');
      }
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is! Map) {
        throw const FormatException('Expected a JSON object');
      }
      final data = objectMap(decoded);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw StateError(
          data['message']?.toString() ?? 'Server error ${response.statusCode}',
        );
      }
      return data;
    } finally {
      timer.cancel();
      _requests.remove(cancel);
    }
  }

  void _abortRequests() {
    for (final request in _requests.toList()) {
      if (!request.isCompleted) request.complete();
    }
  }

  Future<void> pollStatus() async {
    if (_disposed || _pollBusy) return;
    _pollTimer?.cancel();
    _pollBusy = true;
    final generation = _generation;
    try {
      final data = await requestJson('/system-status');
      if (generation == _generation && !_disposed) applyStatus(data);
    } catch (_) {
      if (generation == _generation && !_disposed) {
        _offline('Python server is unavailable');
      }
    } finally {
      _pollBusy = false;
      if (!_disposed) {
        final delay = serverOnline && _foreground
            ? statusPollInterval
            : const Duration(seconds: 2);
        _pollTimer = Timer(delay, () => unawaited(pollStatus()));
      }
    }
  }

  /// Public for deterministic tests. Actual updates arrive through /system-status.
  void applyStatus(Map<String, dynamic> data) {
    if (_disposed) return;
    serverOnline = true;
    _lastStatusAt = _clock();
    normal = objectMap(data['normal']);
    camera = objectMap(data['camera']);
    performance = objectMap(data['performance']);
    gemini = objectMap(data['gemini']);
    explore = objectMap(data['explore']);
    connectionStatus = cameraLive
        ? 'Camera connected'
        : 'Server online · camera waiting';
    guidance = normal['message']?.toString().trim() ?? 'Waiting for guidance.';
    lastError = performance['worker_error']?.toString();
    final status = normal['status']?.toString();
    // Never announce an old outage after a newer status has replaced it.
    if (status == 'ok' || exploring || sosActive) feedback.remove('connection');
    if (cameraLive && status == 'ok' && !exploring && !sosActive) {
      _lastConnectionNotice = null;
      final generation = _generation;
      final received = _clock();
      feedback.submit(
        VoiceJob(
          channel: 'guidance',
          english: guidance,
          isValid: () =>
          !_disposed &&
              generation == _generation &&
              cameraLive &&
              normal['status'] == 'ok' &&
              !exploring &&
              !sosActive &&
              _clock() - received < const Duration(seconds: 4),
        ),
      );
    } else {
      feedback.remove('guidance');
      if (!exploring &&
          !sosActive &&
          status != null &&
          status != _lastConnectionNotice) {
        _lastConnectionNotice = status;
        _localJob(
          'connection',
          guidance,
          'নতুন তথ্যের জন্য অপেক্ষা করা হচ্ছে।',
          priority: 1,
          valid: () =>
          freshConnection &&
              normal['status'] == status &&
              !exploring &&
              !sosActive,
        );
      }
    }
    // Polling and SMS do not wait for speech, even during Explore.
    if (sosActive) {
      unawaited(_handleSos());
      if (_lastSosReminder == null ||
          _clock() - _lastSosReminder! >= const Duration(seconds: 30)) {
        _lastSosReminder = _clock();
        _localJob(
          'sos',
          'Emergency S O S is active. Please check the app.',
          'জরুরি এসওএস সক্রিয় আছে। অনুগ্রহ করে অ্যাপটি দেখুন।',
          priority: 5,
          valid: () => sosActive,
        );
      }
    } else {
      _lastSosReminder = null;
      feedback.remove('sos');
    }
    _consumeExplore();
    _notify();
  }

  void _offline(String message) {
    serverOnline = false;
    connectionStatus = message;
    guidance = 'Connection lost. Live guidance is unavailable.';
    feedback.remove('guidance');
    if (_lastConnectionNotice != 'offline') {
      _lastConnectionNotice = 'offline';
      _localJob(
        'connection',
        guidance,
        'সংযোগ বিচ্ছিন্ন হয়েছে। নতুন নির্দেশনা পাওয়া যাচ্ছে না।',
        priority: 1,
        valid: () => !freshConnection,
      );
    }
    _notify();
  }

  void _localJob(
      String channel,
      String english,
      String bangla, {
        int priority = 1,
        bool Function()? valid,
      }) {
    final generation = _generation;
    feedback.submit(
      VoiceJob(
        channel: channel,
        english: english,
        bangla: bangla,
        priority: priority,
        isValid: () =>
        !_disposed && generation == _generation && (valid?.call() ?? true),
      ),
    );
  }

  Future<void> _speakJob(VoiceJob job) async {
    if (!voiceEnabled || !_ttsReady || _disposed || !job.valid) return;
    final generation = _speechGeneration;
    final selectedLanguage = language;
    String? bangla = job.bangla;
    try {
      if (selectedLanguage != 'en') {
        if (!_banglaAvailable) {
          throw StateError(
            'Install a Bangla voice in the phone’s text-to-speech settings.',
          );
        }
        bangla ??= _translations[job.english];
        if (bangla == null) {
          final result = await _translator
              .translate(job.english, from: 'en', to: 'bn')
              .timeout(const Duration(seconds: 5));
          bangla = result.text.trim();
          if (bangla.isEmpty) {
            throw StateError('Bangla translation is unavailable');
          }
          _translations[job.english] = bangla;
          if (_translations.length > 48) {
            _translations.remove(_translations.keys.first);
          }
        }
      }
      if (_disposed ||
          generation != _speechGeneration ||
          !voiceEnabled ||
          !job.valid) {
        return;
      }
      _voiceRestart?.cancel();
      if (_speechAvailable) await _speech.cancel();
      microphoneListening = false;
      _notify();
      if (selectedLanguage != 'bn') {
        await _utter(job.english, 'en-US', generation);
      }
      // English + Bangla is ONE feedback message. The 12-second gap begins
      // after both segments, not between the two translations.
      if (selectedLanguage != 'en' &&
          bangla != null &&
          generation == _speechGeneration &&
          voiceEnabled &&
          !_disposed) {
        await _utter(bangla, _banglaLocale, generation);
      }
      lastSpoken = job.english;
      audioStatus = '12 seconds of quiet after every voice message';
    } catch (_) {
      audioStatus = selectedLanguage == 'en'
          ? 'Voice playback failed. Check your speech engine.'
          : 'Bangla speech unavailable. Check the voice pack and internet connection.';
    } finally {
      _notify();
    }
  }

  Future<void> _utter(String text, String locale, int generation) async {
    if (_disposed || generation != _speechGeneration || !voiceEnabled) return;
    final cancel = Completer<void>();
    _stopUtterance = cancel;
    try {
      await _tts.setLanguage(locale);
      if (_disposed || generation != _speechGeneration || !voiceEnabled) return;
      // Await completion, not merely the platform's "speech started" signal.
      await Future.any<void>([
        _tts.speak(text).then<void>((_) {}),
        cancel.future,
      ]).timeout(const Duration(seconds: 90));
    } on TimeoutException {
      await _tts.stop();
      rethrow;
    } finally {
      if (identical(_stopUtterance, cancel)) _stopUtterance = null;
    }
  }

  Future<void> _stopVoice() async {
    _speechGeneration++;
    if (enableDeviceServices) {
      try {
        await _tts.stop();
      } catch (_) {}
    }
    final completion = _stopUtterance;
    if (completion != null && !completion.isCompleted) completion.complete();
  }

  void _onSpeechState() {
    _notify();
    if (!feedback.busy) _scheduleListening();
  }

  Future<void> toggleVoice() async {
    voiceEnabled = !voiceEnabled;
    feedback.suspend(!voiceEnabled || (enableDeviceServices && !_ttsReady));
    if (!voiceEnabled) await _stopVoice();
    await _preferences?.setBool('voice_feedback_enabled', voiceEnabled);
    _notify();
  }

  void repeatGuidance() {
    if (!voiceEnabled || !cameraLive || normal['status'] != 'ok') return;
    final generation = _generation;
    feedback.submit(
      VoiceJob(
        channel: 'replay',
        english: guidance,
        priority: 2,
        isValid: () =>
        generation == _generation &&
            cameraLive &&
            normal['status'] == 'ok' &&
            !exploring,
      ),
    );
    _notify();
  }

  Future<void> startExplore() async {
    if (_disposed || exploring || !cameraLive || gemini['configured'] != true) {
      return;
    }
    exploreStarting = true;
    explorationDescription = null;
    _exploreStartedAt = _clock();
    feedback.remove('guidance');
    feedback.remove('replay');
    _voiceRestart?.cancel();
    if (enableDeviceServices && _speechAvailable) await _speech.cancel();
    _notify();
    final generation = _generation;
    try {
      final response = await requestJson(
        '/command',
        body: {'command': 'explore'},
        timeout: const Duration(seconds: 5),
      );
      if (_disposed || generation != _generation) return;
      _requestedExploreId = response['request_id']?.toString();
      if (_requestedExploreId == null) {
        throw StateError('The server did not return an exploration ID.');
      }
      // Progress is visual. One shared status poll also watches the result,
      // keeping distance and SOS active without a second polling loop.
      explorationDescription = 'Analyzing your surroundings…';
    } catch (error) {
      if (generation != _generation || _disposed) return;
      exploreStarting = false;
      _exploreStartedAt = null;
      explorationDescription = 'Scene exploration could not start.';
      lastError = error.toString();
      _localJob(
        'explore',
        'Scene exploration could not start.',
        'দৃশ্য বিশ্লেষণ শুরু করা যায়নি।',
        priority: 3,
      );
      _scheduleListening();
    }
    _notify();
  }

  void _consumeExplore() {
    final id = explore['request_id']?.toString();
    final status = explore['status'];
    if (id != null &&
        id == _requestedExploreId &&
        id != _handledExploreId &&
        (status == 'ready' || status == 'error')) {
      _handledExploreId = id;
      exploreStarting = false;
      _exploreStartedAt = null;
      explorationDescription =
          (explore['description'] ??
              explore['message'] ??
              'Scene description unavailable.')
              .toString();
      final generation = _generation;
      final queuedAt = _clock();
      feedback.submit(
        VoiceJob(
          channel: 'explore',
          english: explorationDescription!,
          priority: 3,
          isValid: () =>
          generation == _generation &&
              _requestedExploreId == id &&
              _clock() - queuedAt < const Duration(seconds: 60),
        ),
      );
      _scheduleListening();
    } else if (_exploreStartedAt != null &&
        _clock() - _exploreStartedAt! > const Duration(seconds: 90)) {
      exploreStarting = false;
      _requestedExploreId = null;
      _exploreStartedAt = null;
      explorationDescription =
      'Scene exploration timed out. Check server status.';
      _localJob(
        'explore',
        explorationDescription!,
        'দৃশ্য বিশ্লেষণের সময় শেষ হয়েছে। সার্ভারের অবস্থা দেখুন।',
        priority: 3,
      );
    }
  }

  Future<void> _handleSos() async {
    if (_handlingSos || _disposed) return;
    final event = normal['sos_event_id'];
    final stamp = normal['sos_last_pressed_at'];
    if (event == null || stamp == null) return;
    final fingerprint = '$server|$stamp|$event';
    if (_handledSos.contains(fingerprint)) return;
    _handlingSos = true;
    _handledSos.add(fingerprint);
    if (_handledSos.length > 40) _handledSos.remove(_handledSos.first);
    final number = emergencyNumber.trim();
    final pos = position;
    try {
      // Persist before asking the native bridge to send: polling/reconnect/app
      // relaunch must not repeatedly send the same emergency text.
      await _preferences?.setStringList(
        'handled_sos_events_v2',
        _handledSos.toList(),
      );
      if (_disposed) return;
      if (number.isEmpty) {
        smsStatus = 'No emergency number saved. Add one in Settings.';
        _localJob(
          'sms',
          'No emergency phone number is saved. Please check settings.',
          'জরুরি ফোন নম্বর সংরক্ষণ করা হয়নি। সেটিংস দেখুন।',
          priority: 4,
        );
        return;
      }
      smsStatus = 'Preparing the emergency message…';
      _notify();
      final location = pos == null
          ? 'location unavailable'
          : 'https://maps.google.com/?q=${pos.latitude},${pos.longitude} (fix: ${pos.timestamp.toIso8601String()})';
      final body = 'SOS! I need help. My last known location: $location';
      if (smsOutput != null) {
        await smsOutput!(number, body);
        smsStatus = 'Emergency message submitted';
      } else if (enableDeviceServices &&
          !kIsWeb &&
          defaultTargetPlatform == TargetPlatform.android) {
        final allowed =
            await _smsChannel.invokeMethod<bool>('requestSmsPermission') ??
                false;
        if (!allowed) throw StateError('SMS permission was not granted');
        if (_disposed) return;
        final sent =
            await _smsChannel.invokeMethod<bool>('sendSms', {
              'phone': number,
              'message': body,
            }) ??
                false;
        if (!sent) throw StateError('Android did not accept the message');
        // The existing native bridge reports submission, not delivery receipts.
        smsStatus = 'Submitted to Android · delivery is not confirmed';
      } else if (enableDeviceServices) {
        final uri = Uri(
          scheme: 'sms',
          path: number,
          queryParameters: {'body': body},
        );
        if (!await launchUrl(uri)) {
          throw StateError('SMS composer could not open');
        }
        smsStatus = 'SMS composer opened · tap Send to finish';
      } else {
        smsStatus = 'Device services disabled';
      }
    } catch (_) {
      smsStatus = 'Message not confirmed. Check SMS permission and your phone.';
      _localJob(
        'sms',
        'The emergency message could not be confirmed. Please check your phone.',
        'জরুরি বার্তা নিশ্চিত করা যায়নি। অনুগ্রহ করে ফোনটি দেখুন।',
        priority: 4,
      );
    } finally {
      _handlingSos = false;
      _notify();
    }
  }

  Future<void> acknowledgeSos() async {
    if (!freshConnection || !sosActive || sosPressed || acknowledging) return;
    acknowledging = true;
    _notify();
    try {
      await requestJson('/sos/ack', body: {'event_id': normal['sos_event_id']});
      await pollStatus();
    } catch (error) {
      lastError = error.toString();
    } finally {
      acknowledging = false;
      _notify();
    }
  }

  Future<void> _initializeGps() async {
    try {
      if (!await Geolocator.isLocationServiceEnabled()) {
        gpsStatus = 'Turn on phone location';
        return;
      }
      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (_disposed) return;
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        gpsStatus = 'Location permission is unavailable';
        return;
      }
      const settings = LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 2,
      );
      gpsStatus = 'Finding your location…';
      // Subscribe first; a slow initial fix cannot hold up the rest of startup.
      _positionSubscription =
          Geolocator.getPositionStream(locationSettings: settings).listen(
                (value) {
              if (_disposed) return;
              position = value;
              gpsStatus = 'Location active';
              _notify();
            },
            onError: (_) {
              gpsStatus = 'Location signal unavailable';
              _notify();
            },
          );
      unawaited(
        Geolocator.getCurrentPosition(locationSettings: settings)
            .timeout(const Duration(seconds: 12))
            .then<void>((value) {
          if (_disposed) return;
          if (position == null ||
              value.timestamp.isAfter(position!.timestamp)) {
            position = value;
          }
          gpsStatus = 'Location active';
          _notify();
          unawaited(uploadGps());
        })
            .catchError((Object _) {}),
      );
    } catch (_) {
      gpsStatus = 'Location could not start';
    } finally {
      _notify();
    }
  }

  Future<void> uploadGps() async {
    final pos = position;
    if (_disposed || _gpsBusy || pos == null) return;
    // Do not keep making an old fix look fresh on the server.
    if (DateTime.now().difference(pos.timestamp) >
        const Duration(seconds: 30)) {
      gpsStatus = 'Last known location · waiting for a fresh fix';
      _notify();
      return;
    }
    _gpsBusy = true;
    try {
      await requestJson(
        '/gps',
        body: {
          'latitude': pos.latitude,
          'longitude': pos.longitude,
          'accuracy': pos.accuracy,
          'speed': pos.speed < 0 ? 0 : pos.speed,
          'heading': pos.heading,
          'altitude': pos.altitude,
          'timestamp': pos.timestamp.millisecondsSinceEpoch,
        },
      );
      if (_disposed) return;
      lastGpsUpload = DateTime.now();
      gpsStatus = 'Location shared with your server';
    } catch (_) {
      if (!_disposed) gpsStatus = 'Location available · upload waiting';
    } finally {
      _gpsBusy = false;
      _notify();
    }
  }

  Future<void> _initializeSpeech() async {
    try {
      _speechAvailable = await _speech.initialize(
        onStatus: (value) {
          if (_disposed) return;
          microphoneListening = value == 'listening';
          if (!microphoneListening) _scheduleListening();
          _notify();
        },
        onError: (_) {
          microphoneListening = false;
          _scheduleListening(const Duration(seconds: 3));
          _notify();
        },
      );
      _scheduleListening();
    } catch (_) {
      _speechAvailable = false;
    }
    _notify();
  }

  void _scheduleListening([
    Duration delay = const Duration(milliseconds: 900),
  ]) {
    _voiceRestart?.cancel();
    if (_disposed ||
        !enableDeviceServices ||
        !_foreground ||
        !handsFree ||
        !_speechAvailable ||
        feedback.busy ||
        exploring) {
      return;
    }
    _voiceRestart = Timer(delay, () => unawaited(_startListening()));
  }

  Future<void> _startListening() async {
    if (_disposed ||
        !handsFree ||
        !_foreground ||
        !_speechAvailable ||
        _startingMic ||
        feedback.busy ||
        exploring ||
        _speech.isListening) {
      return;
    }
    _startingMic = true;
    try {
      await _speech.listen(
        onResult: _onSpeechResult,
        listenOptions: stt.SpeechListenOptions(
          localeId: 'en_US',
          listenFor: const Duration(seconds: 30),
          pauseFor: const Duration(seconds: 3),
          partialResults: true,
          cancelOnError: true,
        ),
      );
      // A voice slot may have started while the OS opened the microphone.
      if (_disposed ||
          feedback.busy ||
          exploring ||
          !handsFree ||
          !_foreground) {
        await _speech.cancel();
      }
      microphoneListening = _speech.isListening;
    } catch (_) {
      _scheduleListening(const Duration(seconds: 3));
    } finally {
      _startingMic = false;
      _notify();
    }
  }

  void _onSpeechResult(SpeechRecognitionResult result) {
    if (_disposed || feedback.busy || !handsFree || exploring) return;
    recognizedSpeech = result.recognizedWords.trim();
    if (RegExp(
      r'\bexplore\b',
      caseSensitive: false,
    ).hasMatch(recognizedSpeech)) {
      unawaited(startExplore());
    }
    _notify();
  }

  void onLifecycle(AppLifecycleState state) {
    _foreground = state == AppLifecycleState.resumed;
    if (_foreground) {
      unawaited(pollStatus());
      _scheduleListening();
    } else {
      _voiceRestart?.cancel();
      if (enableDeviceServices && _speechAvailable) unawaited(_speech.cancel());
      microphoneListening = false;
    }
    // Preserve best-effort background guidance/GPS. This is not a foreground
    // service: mobile OS suspension can still stop Dart timers in background.
  }

  Future<void> saveSettings(AppSettings settings) async {
    final normalized = normalizeServerAddress(settings.server);
    if (normalized == null) {
      throw const FormatException('Enter a valid server IP or URL.');
    }
    final changed = normalized != server;
    _generation++;
    _abortRequests();
    feedback.clear();
    await _stopVoice();
    if (_disposed) return;
    server = normalized;
    language = settings.language;
    emergencyNumber = settings.emergencyNumber.trim();
    handsFree = settings.handsFree;
    _requestedExploreId = null;
    _exploreStartedAt = null;
    exploreStarting = false;
    if (changed) {
      serverOnline = false;
      normal = {};
      camera = {};
      performance = {};
      gemini = {};
      explore = {};
      guidance = 'Connecting to the updated server…';
      explorationDescription = null;
      _lastConnectionNotice = null;
    }
    await _preferences?.setString('python_server_base_url', server);
    await _preferences?.setString('audio_language', language);
    await _preferences?.setString('sos_phone_number', emergencyNumber);
    await _preferences?.setBool('hands_free_enabled', handsFree);
    if (!handsFree && enableDeviceServices && _speechAvailable) {
      await _speech.cancel();
    }
    _scheduleListening();
    unawaited(pollStatus());
    _notify();
  }

  Future<void> openDashboard() async {
    if (!await launchUrl(
      Uri.parse('$server/dashboard'),
      mode: LaunchMode.externalApplication,
    )) {
      lastError = 'Could not open the dashboard.';
      _notify();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _generation++;
    _pollTimer?.cancel();
    _gpsTimer?.cancel();
    _voiceRestart?.cancel();
    _clockTimer?.cancel();
    _abortRequests();
    _client.close();
    unawaited(_positionSubscription?.cancel() ?? Future<void>.value());
    feedback.removeListener(_onSpeechState);
    feedback.dispose();
    unawaited(_stopVoice());
    if (enableDeviceServices && _speechAvailable) unawaited(_speech.cancel());
    super.dispose();
  }
}

class AppColors {
  static const background = Color(0xFF0B1320);
  static const surface = Color(0xFF142131);
  static const raised = Color(0xFF1C2C40);
  static const line = Color(0xFF2A3B50);
  static const text = Color(0xFFEDF3F9);
  static const muted = Color(0xFFB0BED0);
  static const teal = Color(0xFF4CE0BF);
  static const blue = Color(0xFF9BC1FF);
  static const amber = Color(0xFFF4CB78);
  static const red = Color(0xFFFF8B99);
}

class BlindAssistantApp extends StatelessWidget {
  const BlindAssistantApp({super.key, this.controller, this.initialize = true});
  final AssistantController? controller;
  final bool initialize;

  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'Blind Assistant',
    theme: ThemeData(
      useMaterial3: true,
      splashFactory: InkRipple.splashFactory,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.background,
      colorScheme: ColorScheme.fromSeed(
        seedColor: AppColors.teal,
        brightness: Brightness.dark,
        surface: AppColors.surface,
        primary: AppColors.teal,
        onPrimary: const Color(0xFF092B23),
      ),
      textTheme: ThemeData.dark().textTheme.apply(
        bodyColor: AppColors.text,
        displayColor: AppColors.text,
      ),
      inputDecorationTheme: const InputDecorationTheme(
        filled: true,
        fillColor: AppColors.background,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.all(Radius.circular(14)),
        ),
      ),
      snackBarTheme: const SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.background,
        surfaceTintColor: Colors.transparent,
      ),
    ),
    home: BlindAssistantHome(controller: controller, initialize: initialize),
  );
}

class BlindAssistantHome extends StatefulWidget {
  const BlindAssistantHome({
    super.key,
    this.controller,
    this.initialize = true,
  });
  final AssistantController? controller;
  final bool initialize;
  @override
  State<BlindAssistantHome> createState() => _BlindAssistantHomeState();
}

class _BlindAssistantHomeState extends State<BlindAssistantHome>
    with WidgetsBindingObserver {
  late final AssistantController c;

  @override
  void initState() {
    super.initState();
    c = widget.controller ?? AssistantController();
    WidgetsBinding.instance.addObserver(this);
    if (widget.initialize) unawaited(c.initialize());
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) =>
      c.onLifecycle(state);

  Future<void> _settings() async {
    final result = await showModalBottomSheet<AppSettings>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      builder: (_) => SettingsSheet(
        initial: AppSettings(
          server: c.server,
          language: c.language,
          emergencyNumber: c.emergencyNumber,
          handsFree: c.handsFree,
        ),
      ),
    );
    if (result == null || !mounted) return;
    try {
      await c.saveSettings(result);
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Settings saved.')));
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('$error')));
    }
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: c,
    builder: (context, _) {
      return Scaffold(
        appBar: AppBar(
          titleSpacing: 22,
          title: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.explore_rounded, color: AppColors.teal, size: 29),
              SizedBox(width: 10),
              Flexible(
                child: Text(
                  'Wayfinder',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    letterSpacing: -.5,
                  ),
                ),
              ),
            ],
          ),
          actions: [
            IconButton(
              tooltip: 'Settings',
              onPressed: _settings,
              icon: const Icon(Icons.tune_rounded),
            ),
            const SizedBox(width: 10),
          ],
        ),
        body: SafeArea(
          child: Align(
            alignment: Alignment.topCenter,
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 850),
              child: RefreshIndicator(
                onRefresh: c.pollStatus,
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(22, 10, 22, 30),
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            'YOUR ASSISTANCE COMPANION',
                            style: TextStyle(
                              color: AppColors.muted,
                              fontSize: 10,
                              letterSpacing: 1.9,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        StatusPill(
                          label: c.cameraLive
                              ? 'Connected'
                              : c.serverOnline
                              ? 'Camera waiting'
                              : 'Offline',
                          color: c.cameraLive
                              ? AppColors.teal
                              : AppColors.amber,
                        ),
                      ],
                    ),
                    const SizedBox(height: 18),
                    const Text(
                      'Your surroundings.',
                      style: TextStyle(
                        fontSize: 29,
                        height: 1.2,
                        fontWeight: FontWeight.w700,
                        letterSpacing: -.8,
                      ),
                    ),
                    const SizedBox(height: 7),
                    const Text(
                      'Live guidance, with time to listen.',
                      style: TextStyle(color: AppColors.muted, fontSize: 14),
                    ),
                    const SizedBox(height: 22),
                    if (c.sosActive) ...[
                      _sosCard(),
                      const SizedBox(height: 16),
                    ],
                    _guidanceCard(),
                    const SizedBox(height: 16),
                    _voiceCard(),
                    const SizedBox(height: 16),
                    LayoutBuilder(
                      builder: (context, box) {
                        final cards = [_distanceCard(), _objectsCard()];
                        if (box.maxWidth < 340 ||
                            MediaQuery.textScalerOf(context).scale(1) > 1.3) {
                          return Column(
                            children: [
                              cards[0],
                              const SizedBox(height: 14),
                              cards[1],
                            ],
                          );
                        }
                        return IntrinsicHeight(
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Expanded(child: cards[0]),
                              const SizedBox(width: 14),
                              Expanded(child: cards[1]),
                            ],
                          ),
                        );
                      },
                    ),
                    const SizedBox(height: 16),
                    _exploreCard(),
                    const SizedBox(height: 22),
                    _sectionTitle(
                      'In this view',
                      Icons.center_focus_strong_rounded,
                    ),
                    const SizedBox(height: 12),
                    _detectionsCard(),
                    const SizedBox(height: 22),
                    _sectionTitle('Your connection', Icons.sensors_rounded),
                    const SizedBox(height: 12),
                    _connectionCard(),
                    const SizedBox(height: 16),
                    if (c.lastError != null)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 14),
                        child: Text(
                          c.lastError!,
                          style: const TextStyle(
                            color: AppColors.amber,
                            fontSize: 12,
                          ),
                        ),
                      ),
                    Center(
                      child: Text(
                        'ESP32-S3  •  ${c.languageLabel}  •  12-second voice gap',
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontSize: 11,
                          color: AppColors.muted,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
    },
  );

  Widget _sectionTitle(String title, IconData icon) => Row(
    children: [
      Icon(icon, size: 18, color: AppColors.muted),
      const SizedBox(width: 9),
      Expanded(
        child: Text(
          title,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
    ],
  );

  Widget _guidanceCard() => SurfaceCard(
    color: const Color(0xFF163632),
    border: const Color(0xFF2C5650),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Expanded(
              child: Text(
                'CURRENT GUIDANCE',
                style: TextStyle(
                  fontSize: 10,
                  letterSpacing: 1.7,
                  color: AppColors.teal,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            IconButton(
              tooltip: 'Queue this guidance for the next voice slot',
              onPressed:
              c.voiceEnabled && c.cameraLive && c.normal['status'] == 'ok'
                  ? c.repeatGuidance
                  : null,
              icon: const Icon(Icons.volume_up_rounded, color: AppColors.teal),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          c.guidance,
          style: const TextStyle(
            fontSize: 22,
            height: 1.5,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 18),
        Row(
          children: [
            Icon(
              c.cameraLive
                  ? Icons.check_circle_outline_rounded
                  : Icons.hourglass_empty_rounded,
              size: 14,
              color: const Color(0xFFACD5CA),
            ),
            const SizedBox(width: 7),
            Expanded(
              child: Text(
                c.exploring
                    ? 'Exploring · distance and SOS stay active'
                    : c.cameraLive
                    ? 'Latest detection + live distance'
                    : 'Waiting for fresh information',
                style: const TextStyle(color: Color(0xFFACD5CA), fontSize: 11),
              ),
            ),
          ],
        ),
      ],
    ),
  );

  Widget _voiceCard() => SurfaceCard(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppColors.raised,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                c.feedback.busy
                    ? Icons.graphic_eq_rounded
                    : Icons.timer_outlined,
                color: AppColors.teal,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    c.voiceStatus,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'A 12s pause after each message',
                    style: TextStyle(fontSize: 11, color: AppColors.muted),
                  ),
                ],
              ),
            ),
            IconButton(
              tooltip: c.voiceEnabled
                  ? 'Pause all voice feedback'
                  : 'Resume voice feedback',
              onPressed: () => unawaited(c.toggleVoice()),
              icon: Icon(
                c.voiceEnabled
                    ? Icons.pause_circle_outline_rounded
                    : Icons.play_circle_outline_rounded,
                color: AppColors.teal,
                size: 30,
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        LinearProgressIndicator(
          value: c.feedback.busy ? null : c.quietSeconds / 12,
          minHeight: 4,
          borderRadius: BorderRadius.circular(4),
          color: AppColors.teal,
          backgroundColor: AppColors.line,
        ),
        if (c.audioStatus.contains('unavailable') ||
            c.audioStatus.contains('failed') ||
            c.audioStatus.contains('could not'))
          Padding(
            padding: const EdgeInsets.only(top: 10),
            child: Text(
              c.audioStatus,
              style: const TextStyle(color: AppColors.amber, fontSize: 12),
            ),
          ),
      ],
    ),
  );

  Widget _distanceCard() {
    final distance = c.distance;
    final status = c.normal['distance_status'];
    final color = distance == null
        ? AppColors.muted
        : distance <= 50
        ? AppColors.red
        : distance <= 100
        ? AppColors.amber
        : AppColors.teal;
    return SurfaceCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.radar_rounded, color: AppColors.teal, size: 18),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Distance ahead',
                  style: TextStyle(color: AppColors.muted, fontSize: 12),
                ),
              ),
            ],
          ),
          const SizedBox(height: 15),
          Text.rich(
            TextSpan(
              children: [
                TextSpan(
                  text: distance?.toStringAsFixed(0) ?? '—',
                  style: const TextStyle(
                    fontSize: 36,
                    fontWeight: FontWeight.w600,
                    letterSpacing: -1,
                  ),
                ),
                const TextSpan(
                  text: ' cm',
                  style: TextStyle(fontSize: 13, color: AppColors.muted),
                ),
              ],
            ),
          ),
          const SizedBox(height: 9),
          Text(
            distance == null
                ? (status == 'sensor_error'
                ? 'No echo received'
                : 'No fresh reading')
                : distance <= 30
                ? 'Very close obstacle'
                : distance <= 50
                ? 'Close obstacle'
                : distance <= 100
                ? 'Within one meter'
                : 'Beyond one meter',
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _objectsCard() => SurfaceCard(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(
          children: [
            Icon(Icons.view_in_ar_rounded, color: AppColors.blue, size: 18),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                'Objects in view',
                style: TextStyle(color: AppColors.muted, fontSize: 12),
              ),
            ),
          ],
        ),
        const SizedBox(height: 15),
        Text(
          c.cameraLive && c.normal['status'] == 'ok'
              ? '${c.detections.length}'
              : '—',
          style: const TextStyle(fontSize: 36, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 9),
        Text(
          c.exploring
              ? 'Exploring the scene'
              : c.cameraLive
              ? 'Latest camera result'
              : 'Waiting for camera',
          style: const TextStyle(color: AppColors.muted, fontSize: 11),
        ),
      ],
    ),
  );

  Widget _exploreCard() => SurfaceCard(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(
          children: [
            Expanded(
              child: Text(
                'Explore the scene',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
              ),
            ),
            Icon(Icons.auto_awesome_rounded, color: AppColors.blue, size: 21),
          ],
        ),
        const SizedBox(height: 9),
        Text(
          c.exploring
              ? 'Understanding your surroundings. Sensors stay active.'
              : 'Tap below or say “explore” for a spoken scene description.',
          style: const TextStyle(
            color: AppColors.muted,
            fontSize: 13,
            height: 1.6,
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          child: FilledButton.icon(
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(13),
              ),
            ),
            onPressed:
            c.cameraLive && c.gemini['configured'] == true && !c.exploring
                ? () => unawaited(c.startExplore())
                : null,
            icon: c.exploring
                ? const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
                : const Icon(Icons.auto_awesome_outlined, size: 20),
            label: Text(
              c.exploring ? 'Exploring…' : 'Describe surroundings',
              textAlign: TextAlign.center,
            ),
          ),
        ),
        const SizedBox(height: 11),
        Row(
          children: [
            Icon(
              c.microphoneListening
                  ? Icons.mic_rounded
                  : Icons.mic_none_rounded,
              size: 15,
              color: c.microphoneListening ? AppColors.teal : AppColors.muted,
            ),
            const SizedBox(width: 7),
            Expanded(
              child: Text(
                !c.handsFree
                    ? 'Hands-free listening is off'
                    : c.feedback.busy
                    ? 'Microphone rests during speech'
                    : c.microphoneListening
                    ? 'Listening for “explore”'
                    : 'Tap to explore · voice listening on standby',
                style: const TextStyle(fontSize: 11, color: AppColors.muted),
              ),
            ),
          ],
        ),
        if (c.explorationDescription != null) ...[
          const Divider(height: 28, color: AppColors.line),
          Text(
            c.explorationDescription!,
            style: const TextStyle(fontSize: 15, height: 1.6),
          ),
        ],
      ],
    ),
  );

  Widget _sosCard() => SurfaceCard(
    color: const Color(0xFF402430),
    border: const Color(0xFF754052),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(
          children: [
            Icon(Icons.sos_rounded, color: AppColors.red, size: 30),
            SizedBox(width: 10),
            Expanded(
              child: Text(
                'SOS needs attention',
                style: TextStyle(
                  color: AppColors.red,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Text(c.smsStatus, style: const TextStyle(fontSize: 13, height: 1.5)),
        const SizedBox(height: 7),
        const Text(
          'Voice alerts also respect the 12-second quiet period.',
          style: TextStyle(color: Color(0xFFE4BBC5), fontSize: 11),
        ),
        const SizedBox(height: 13),
        OutlinedButton(
          onPressed: !c.freshConnection || c.sosPressed || c.acknowledging
              ? null
              : () => unawaited(c.acknowledgeSos()),
          child: Text(
            c.sosPressed
                ? 'Release the SOS button first'
                : c.acknowledging
                ? 'Acknowledging…'
                : 'Acknowledge SOS',
          ),
        ),
      ],
    ),
  );

  Widget _detectionsCard() {
    final items = c.detections;
    return SurfaceCard(
      child: items.isEmpty
          ? Row(
        children: [
          const Icon(
            Icons.center_focus_weak_rounded,
            color: AppColors.muted,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              c.cameraLive && c.normal['status'] == 'ok'
                  ? 'No recognized objects in this frame.'
                  : 'Waiting for a fresh detection result.',
              style: const TextStyle(
                color: AppColors.muted,
                fontSize: 13,
              ),
            ),
          ),
        ],
      )
          : Column(
        children: [
          for (var i = 0; i < items.length; i++) ...[
            if (i > 0) const Divider(height: 24, color: AppColors.line),
            Wrap(
              alignment: WrapAlignment.spaceBetween,
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 12,
              runSpacing: 8,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        color: AppColors.teal,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Flexible(
                      child: Text(
                        '${items[i]['class']}',
                        style: const TextStyle(
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
                StatusPill(
                  label: switch (items[i]['direction']) {
                    'left' => 'Left',
                    'right' => 'Right',
                    _ => 'Ahead',
                  },
                  color: AppColors.blue,
                ),
                Text(
                  '${((finiteNumber(items[i]['confidence']) ?? 0) * 100).round()}%',
                  style: const TextStyle(
                    color: AppColors.muted,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _connectionCard() => SurfaceCard(
    child: Column(
      children: [
        _detailRow(
          Icons.sensors_rounded,
          'Camera & server',
          c.connectionStatus,
          c.cameraLive ? AppColors.teal : AppColors.amber,
        ),
        const Divider(height: 28, color: AppColors.line),
        _detailRow(
          Icons.location_on_outlined,
          'Phone location',
          c.gpsStatus,
          AppColors.blue,
        ),
        const Divider(height: 28, color: AppColors.line),
        _detailRow(
          Icons.shield_outlined,
          'Emergency contact',
          c.emergencyNumber.isEmpty
              ? 'Add a number in Settings'
              : 'Number saved',
          AppColors.muted,
        ),
        const SizedBox(height: 14),
        Wrap(
          spacing: 10,
          runSpacing: 4,
          children: [
            TextButton.icon(
              onPressed: () => unawaited(c.openDashboard()),
              icon: const Icon(Icons.open_in_new_rounded, size: 16),
              label: const Text('Camera dashboard'),
            ),
            TextButton.icon(
              onPressed: _settings,
              icon: const Icon(Icons.tune_rounded, size: 16),
              label: const Text('Settings'),
            ),
          ],
        ),
      ],
    ),
  );

  Widget _detailRow(IconData icon, String label, String value, Color color) =>
      Row(
        children: [
          Icon(icon, size: 23, color: color),
          const SizedBox(width: 13),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: const TextStyle(fontSize: 12, color: AppColors.muted),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ],
      );

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    if (widget.controller == null) c.dispose();
    super.dispose();
  }
}

class SurfaceCard extends StatelessWidget {
  const SurfaceCard({
    super.key,
    required this.child,
    this.color = AppColors.surface,
    this.border = AppColors.line,
  });
  final Widget child;
  final Color color, border;
  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(19),
    decoration: BoxDecoration(
      color: color,
      border: Border.all(color: border),
      borderRadius: BorderRadius.circular(19),
    ),
    child: child,
  );
}

class StatusPill extends StatelessWidget {
  const StatusPill({super.key, required this.label, required this.color});
  final String label;
  final Color color;
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
    decoration: BoxDecoration(
      color: color.withValues(alpha: .1),
      borderRadius: BorderRadius.circular(30),
      border: Border.all(color: color.withValues(alpha: .24)),
    ),
    child: Text(
      label,
      style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w600),
    ),
  );
}

class SettingsSheet extends StatefulWidget {
  const SettingsSheet({super.key, required this.initial});
  final AppSettings initial;
  @override
  State<SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<SettingsSheet> {
  final form = GlobalKey<FormState>();
  late final TextEditingController serverInput, phoneInput;
  late String language;
  late bool handsFree;
  @override
  void initState() {
    super.initState();
    serverInput = TextEditingController(text: widget.initial.server);
    phoneInput = TextEditingController(text: widget.initial.emergencyNumber);
    language = widget.initial.language;
    handsFree = widget.initial.handsFree;
  }

  @override
  Widget build(BuildContext context) => SingleChildScrollView(
    padding: EdgeInsets.fromLTRB(
      24,
      8,
      24,
      24 + MediaQuery.viewInsetsOf(context).bottom,
    ),
    child: Form(
      key: form,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Make it yours',
            style: TextStyle(fontSize: 25, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          const Text(
            'Connection, language, and emergency settings.',
            style: TextStyle(color: AppColors.muted),
          ),
          const SizedBox(height: 24),
          TextFormField(
            controller: serverInput,
            keyboardType: TextInputType.url,
            autocorrect: false,
            decoration: const InputDecoration(
              labelText: 'Python server IP or URL',
              hintText: '192.168.0.148:5000',
              prefixIcon: Icon(Icons.dns_outlined),
            ),
            validator: (v) => normalizeServerAddress(v ?? '') == null
                ? 'Enter an IP or http(s) URL without a path.'
                : null,
          ),
          const SizedBox(height: 18),
          TextFormField(
            controller: phoneInput,
            keyboardType: TextInputType.phone,
            decoration: const InputDecoration(
              labelText: 'Emergency SMS number',
              hintText: '+880…',
              prefixIcon: Icon(Icons.phone_outlined),
            ),
            validator: (v) =>
            v != null &&
                v.trim().isNotEmpty &&
                !RegExp(r'^\+?[0-9 ()-]{6,22}$').hasMatch(v.trim())
                ? 'Enter a valid phone number, or leave it blank.'
                : null,
          ),
          const SizedBox(height: 10),
          const Text(
            'A received SOS event can send your last known location to this number.',
            style: TextStyle(fontSize: 12, color: AppColors.muted),
          ),
          const SizedBox(height: 20),
          DropdownButtonFormField<String>(
            initialValue: language,
            isExpanded: true,
            decoration: const InputDecoration(
              labelText: 'Voice language',
              prefixIcon: Icon(Icons.translate_rounded),
            ),
            items: const [
              DropdownMenuItem(
                value: 'en',
                child: Text(
                  'English',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              DropdownMenuItem(
                value: 'bn',
                child: Text(
                  'বাংলা (Bangla)',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              DropdownMenuItem(
                value: 'both',
                child: Text(
                  'English + বাংলা',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
            onChanged: (value) {
              if (value != null) setState(() => language = value);
            },
          ),
          const SizedBox(height: 10),
          SwitchListTile.adaptive(
            contentPadding: EdgeInsets.zero,
            value: handsFree,
            title: const Text('Hands-free “explore”'),
            subtitle: const Text('Microphone pauses while feedback speaks.'),
            onChanged: (value) => setState(() => handsFree = value),
          ),
          const SizedBox(height: 8),
          const SurfaceCard(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.timer_outlined, color: AppColors.teal),
                SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Every voice message is followed by 12 seconds of quiet, including SOS and scene descriptions.',
                    style: TextStyle(fontSize: 13, height: 1.5),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 22),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: () {
                if (form.currentState!.validate()) {
                  Navigator.of(context).pop(
                    AppSettings(
                      server: serverInput.text,
                      language: language,
                      emergencyNumber: phoneInput.text,
                      handsFree: handsFree,
                    ),
                  );
                }
              },
              style: FilledButton.styleFrom(padding: const EdgeInsets.all(16)),
              child: const Text('Save settings'),
            ),
          ),
        ],
      ),
    ),
  );

  @override
  void dispose() {
    serverInput.dispose();
    phoneInput.dispose();
    super.dispose();
  }
}
