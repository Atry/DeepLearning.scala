package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.each.Monadic._
import com.thoughtworks.feature.Factory
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import com.thoughtworks.future._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{AsyncFreeSpec, Inside, Matchers}

import scalaz.syntax.all._
import scalaz.std.iterable._

object ConvolutionalLayersSpec {
  trait learningRate extends com.thoughtworks.deeplearning.plugins.INDArrayWeights {

    import org.nd4s.Implicits._
    import org.nd4j.linalg.api.ndarray.INDArray

    def learningRate: Double

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>

      private lazy val delta0: INDArray = super.delta * learningRate

      override def delta: INDArray = delta0
    }

    override type INDArrayOptimizer <: Optimizer with INDArrayOptimizerApi

  }
  trait Adagrad extends com.thoughtworks.deeplearning.plugins.INDArrayWeights {

    import org.nd4j.linalg.api.ndarray.INDArray
    import org.nd4j.linalg.factory.Nd4j
    import org.nd4j.linalg.ops.transforms.Transforms

    def eps: Double

    trait INDArrayWeightApi extends super.INDArrayWeightApi { this: INDArrayWeight =>
      var cache: Option[INDArray] = None
    }

    override type INDArrayWeight <: INDArrayWeightApi with Weight

    trait INDArrayOptimizerApi extends super.INDArrayOptimizerApi { this: INDArrayOptimizer =>
      private lazy val delta0: INDArray = {
        import org.nd4s.Implicits._
        import weight._
        val superDelta = super.delta
        val newCache = weight.synchronized {
          val newCache = weight.cache.getOrElse(Nd4j.zeros(superDelta.shape: _*)) + superDelta * superDelta
          weight.cache = Some(newCache)
          newCache
        }
        superDelta / (Transforms.sqrt(newCache) + eps)
      }
      override def delta = delta0
    }
    override type INDArrayOptimizer <: INDArrayOptimizerApi with Optimizer
  }
}

/**
  * @author 杨博 (Yang Bo)
  */
final class ConvolutionalLayersSpec extends AsyncFreeSpec with Matchers with Inside {
  import ConvolutionalLayersSpec._

  "Given a multi-layer convolutional network" in {
    val hyperparameters =
      Factory[
        ConvolutionalLayers with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with learningRate]
        .newInstance(learningRate = 0.03)
    import hyperparameters.INDArrayWeight
    import hyperparameters.implicits._

    class ConvolutionalThenRelu(inputDepth: Int, outputDepth: Int,weight: INDArrayWeight,bias: INDArrayWeight) {

      val weight: INDArrayWeight
      val bias: INDArrayWeight

    }

    object ConvolutionalThenRelu{

    }

    val weight1 = {
      import org.nd4s.Implicits._
      val inputDepth = 3
      val outputDepth = 20
      val kernelSize = 3
      INDArrayWeight(Nd4j.randn(outputDepth, inputDepth, 3, 3))
    }
    val bias1 = {
      import org.nd4s.Implicits._
      INDArrayWeight(Nd4j.zeros(20))
    }

    val weight2 = {
      import org.nd4s.Implicits._
      val inputDepth = 3
      val outputDepth = 20
      val kernelSize = 3
      INDArrayWeight(Nd4j.randn(outputDepth, inputDepth, 3, 3))
    }
    val bias2 = {
      import org.nd4s.Implicits._
      INDArrayWeight(Nd4j.zeros(20))
    }

  }

  "INDArray im2col (kernel,stride,padding) --forward" in {
    val hyperparameters =
      Factory[
        ConvolutionalLayers with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with learningRate]
        .newInstance(learningRate = 0.03)
    import hyperparameters.implicits._
    val weight = {
      import org.nd4s.Implicits._
      hyperparameters.INDArrayWeight((-(1 to 54).toNDArray).reshape(2, 3, 3, 3))
    }

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.im2col(weight, kernel, stride, padding)
    }

    myNetwork((3, 3), (1, 1), (1, 1)).train.map { result =>
      import org.nd4s.Implicits._
      result.sumT should be(-8085.0)
    }.toScalaFuture
  }

  "INDArray im2col (kernel,stride,padding) --train" in {
    val hyperparameters =
      Factory[
        ConvolutionalLayers with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with learningRate]
        .newInstance(learningRate = 0.01)
    import hyperparameters.implicits._
    val weight = {
      import org.nd4s.Implicits._
      hyperparameters.INDArrayWeight((1 to 54).toNDArray.reshape(2, 3, 3, 3))
    }

    def myNetwork(kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.im2col(weight, kernel, stride, padding)
    }
    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 1000) {
        myNetwork((3, 3), (1, 1), (1, 1)).train.each
      }
    }

    throwableMonadic[Future] {
      import org.nd4s.Implicits._
      task.each
      val loss = (myNetwork((3, 3), (1, 1), (1, 1))).predict.each
      (loss.sumT: Double) should be < 1.0
    }.toScalaFuture
  }

  "INDArray maxPool poolsize --forward" in {

    val hyperparameters =
      Factory[
        ConvolutionalLayers with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with learningRate]
        .newInstance(learningRate = 1.0)
    import hyperparameters.implicits._

    val weight = {
      import org.nd4s.Implicits._
      hyperparameters.INDArrayWeight((1 to 96).toNDArray.reshape(2, 3, 4, 4))
    }

    def myNetwork(poolSize: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.maxPool(weight, poolSize)
    }

    myNetwork((2, 2)).train.map { result =>
      import org.nd4s.Implicits._
      result.sumT should be(1224.0)
    }.toScalaFuture
  }

  "INDArray maxPool poolsize -- train" in {

    val hyperparameters =
      Factory[
        ConvolutionalLayers with Logging with ImplicitsSingleton with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with learningRate]
        .newInstance(learningRate = 1.0)
    import hyperparameters.implicits._

    val weight = {
      import org.nd4s.Implicits._
      hyperparameters.INDArrayWeight((1 to 96).toNDArray.reshape(2, 3, 4, 4))
    }

    def myNetwork(poolSize: (Int, Int)): hyperparameters.INDArrayLayer = {
      hyperparameters.maxPool(weight, poolSize)
    }

    val poolSize = (2, 2)

    @monadic[Future]
    val task: Future[Unit] = {
      for (_ <- 1 to 700) {
        myNetwork(poolSize).train.each
      }
    }

    throwableMonadic[Future] {
      import org.nd4s.Implicits._
      task.each
      val loss: INDArray = (myNetwork(poolSize)).predict.each
      loss.meanT should be < 10.0
    }.toScalaFuture

  }

}
