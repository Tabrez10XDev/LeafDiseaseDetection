package pro.lj.tflite

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.snackbar.Snackbar
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import pro.lj.tflite.ml.*
import java.lang.String.format
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    val imageSize = 224

    fun onClickRequestPermission(view: View) {
        when {
            ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                Snackbar.make(
                        view,
                        "granted",
                        Snackbar.LENGTH_SHORT
                ).show()
            }

            ActivityCompat.shouldShowRequestPermissionRationale(
                    this,
                    Manifest.permission.CAMERA
            ) -> {
                Snackbar.make(
                        view,
                        "Required",
                        Snackbar.LENGTH_INDEFINITE,
                ).show()
                    requestPermissionLauncher.launch(
                            Manifest.permission.CAMERA
                    )

            }

            else -> {
                requestPermissionLauncher.launch(
                        Manifest.permission.CAMERA
                )
            }
        }
    }

    private val requestPermissionLauncher =
            registerForActivityResult(
                    ActivityResultContracts.RequestPermission()
            ) { isGranted: Boolean ->
                if (isGranted) {
                    Log.d("Permission: ", "Granted")
                } else {
                    Log.d("Permission: ", "Denied")
                }
            }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        onClickRequestPermission(root)
        button.setOnClickListener {

            if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED){
                Log.d("TTT","hello")

                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 1)
            }
        }


    }

    fun classifyImage(image: Bitmap){
        val model = Yourmodel   .newInstance(applicationContext)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // get 1D array of 224 * 224 pixels in image
        val intValues = IntArray(imageSize * imageSize)
        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        Log.d("TTT","1")
        var pixel = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val value = intValues[pixel++] // RGB
                byteBuffer.putFloat((value shr 16 and 0xFF).toFloat())
                byteBuffer.putFloat((value shr 8 and 0xFF).toFloat())
                byteBuffer.putFloat((value and 0xFF).toFloat())
            }
        }
        Log.d("TTT","he")



        inputFeature0.loadBuffer(byteBuffer)
        Log.d("TTT","3")

// Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidences = outputFeature0.floatArray

        var maxPos = 0
        var maxConfidence = 0f
        for (i in 0 until confidences.size) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxPos = i
            }
        }
        val classes = arrayOf("Tomato_Early_blight",
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites_Two_spotted_spider_mite",
                "Tomato__Tomato_YellowLeaf__Curl_Virus",
                "Tomato__Tomato_mosaic_virus",
                "Tomato_healthy")
//        val classes = arrayOf(
//            "Tomato_Early_blight",
//            "Tomato_Septoria_leaf_spot",
//            "Tomato_healthy"
//            )
        result.text = classes[maxPos]

        var s: String? = ""
        for (i in 0 until classes.size) {
            s += format("%s: %.1f%%\n", classes[i], confidences[i] * 100)
        }
        confidence.text = s

// Releases model resources if no longer used.
        model.close()
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if(requestCode == 1 && resultCode == RESULT_OK){
         //   var image : Bitmap ?= data?.extras?.get("data") as Bitmap
            val icon = BitmapFactory.decodeResource(this.resources, R.drawable.heal1)
            var image = icon
               image?.let{
                val dimension = min(it.width, it.height)
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                imageView.setImageBitmap(image)

                image = Bitmap.createScaledBitmap(image!!, imageSize, imageSize, false)
                classifyImage(image!!)
            }
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}