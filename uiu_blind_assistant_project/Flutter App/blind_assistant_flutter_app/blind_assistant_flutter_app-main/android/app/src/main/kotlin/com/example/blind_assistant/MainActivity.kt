package com.example.blind_assistant

import android.Manifest
import android.content.pm.PackageManager
import android.telephony.SmsManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    companion object {
        private const val SMS_CHANNEL = "blind_assistant/sms"
        private const val SMS_PERMISSION_REQUEST = 1001
    }

    private var pendingSmsPermissionResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            SMS_CHANNEL
        ).setMethodCallHandler { call, result ->

            when (call.method) {

                "requestSmsPermission" -> {
                    requestSmsPermission(result)
                }

                "sendSms" -> {
                    val phone = call.argument<String>("phone")?.trim().orEmpty()
                    val message = call.argument<String>("message")?.trim().orEmpty()

                    if (phone.isEmpty()) {
                        result.error(
                            "INVALID_PHONE",
                            "The SOS phone number is empty.",
                            null
                        )
                        return@setMethodCallHandler
                    }

                    if (message.isEmpty()) {
                        result.error(
                            "EMPTY_MESSAGE",
                            "The SMS message is empty.",
                            null
                        )
                        return@setMethodCallHandler
                    }

                    if (
                        ContextCompat.checkSelfPermission(
                            this,
                            Manifest.permission.SEND_SMS
                        ) != PackageManager.PERMISSION_GRANTED
                    ) {
                        result.error(
                            "SMS_PERMISSION_DENIED",
                            "SEND_SMS permission has not been granted.",
                            null
                        )
                        return@setMethodCallHandler
                    }

                    try {
                        sendSmsNative(
                            phone = phone,
                            message = message
                        )

                        result.success(true)
                    } catch (e: SecurityException) {
                        result.error(
                            "SMS_SECURITY_ERROR",
                            e.message ?: "Android blocked SMS sending.",
                            null
                        )
                    } catch (e: Exception) {
                        result.error(
                            "SMS_SEND_ERROR",
                            e.message ?: "Unable to send SMS.",
                            null
                        )
                    }
                }

                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    private fun requestSmsPermission(result: MethodChannel.Result) {
        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.SEND_SMS
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            result.success(true)
            return
        }

        // Avoid having two unresolved Flutter calls waiting for
        // the same Android permission dialog.
        if (pendingSmsPermissionResult != null) {
            result.error(
                "SMS_PERMISSION_PENDING",
                "An SMS permission request is already active.",
                null
            )
            return
        }

        pendingSmsPermissionResult = result

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.SEND_SMS),
            SMS_PERMISSION_REQUEST
        )
    }

    @Suppress("DEPRECATION")
    private fun sendSmsNative(
        phone: String,
        message: String
    ) {
        val smsManager = SmsManager.getDefault()
        val parts = smsManager.divideMessage(message)

        if (parts.size > 1) {
            smsManager.sendMultipartTextMessage(
                phone,
                null,
                parts,
                null,
                null
            )
        } else {
            smsManager.sendTextMessage(
                phone,
                null,
                message,
                null,
                null
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (requestCode != SMS_PERMISSION_REQUEST) {
            return
        }

        val granted =
            grantResults.isNotEmpty() &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED

        pendingSmsPermissionResult?.success(granted)
        pendingSmsPermissionResult = null
    }
}
