package com.thoughtworks.deeplearning.plugins

import java.util.concurrent.Callable

import com.google.common.cache._
import com.thoughtworks.compute.OpenCL
import com.thoughtworks.continuation._
import com.thoughtworks.expressions.api.{Arrays, Floats}
import com.thoughtworks.expressions.opencl.Context
import com.thoughtworks.expressions.opencl.Context.GlobalContext
import com.thoughtworks.expressions.tree.{FloatArrayTrees, StructuralTrees}
import com.thoughtworks.feature.Factory
//import com.thoughtworks.expressions.Anonymous.Implicitly
//import com.thoughtworks.expressions.Builtins.AllOpenCLExpressions
//import com.thoughtworks.expressions.OpenCLValues
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.covariant._
import com.thoughtworks.tryt.covariant._
import org.apache.commons.math3.linear._

import scala.concurrent.ExecutionContext
import scala.language.existentials
import scalaz.std.list._
import scalaz.syntax.all._
import scala.util.Try
import com.dongxiguo.fastring.Fastring.Implicits._

// TODO: Rename to VirtualTensors, like virtual-dom
trait Tensors extends OpenCL {
  trait Closures extends FloatArrayTrees with StructuralTrees {

    type Category = Floats with Arrays

    protected trait TermApi extends super.TermApi {
      this: Term =>
      lazy val upvalues: List[Parameter] = {
//        tree match {
//          case parameter:Parameter =>
//            List(this)
//          case
//        }
//
        ???
      }

    }

    type Term <: TermApi
  }

  val trees: Closures = Factory[Closures].newInstance()
  import trees._

//  final class StructuralKey(val term: Term) {
//
//    override def hashCode(): Int = term.structuralHashCode(Nil).code
//
//    override def equals(that: Any): Boolean = {
//      that match {
//        case that: StructuralKey =>
//          term.isSameStructure(that.term)
//        case _ =>
//          false
//      }
//    }
//  }
//
//  protected trait ExpressionApi extends super.ExpressionApi {
//
//    def isSameStructure(that: Term): Boolean = {
//      structuralComparison(that, Nil, Nil) match {
//        case _: StructuralComparison.Same =>
//          true
//        case StructuralComparison.Different =>
//          false
//      }
//    }
//
//    def structuralComparison(that: Term,
//                             thisParameters: List[Term],
//                             thatParameters: List[Term]): StructuralComparison = {
//      ???
//    }
//
//    def structuralHashCode(thisParameters: List[Term]): StructuralHashCode = {
//      ???
//    }
//
//  }
//
//  type Expression <: ExpressionApi
//

//
//  protected trait TypeApi extends super.TypeApi {
//    this: Type =>
//    @inject
//    def Upvalue: Operator1[Tensor, Identifier with Upvalue]
//  }
//
//  /** @template */
//  type Type <: (Expression with Any) with TypeApi

  // The scalar data type is hard-coded Float at the moment. FIXME: Allow other types in the future
  trait PendingBuffer {
    def event: Event

    def buffer: DeviceBuffer[Float]
  }

  sealed trait Tensor {
    thisTensor =>
//    def debuggingInformation: Implicitly[DebuggingInformation]
//
    def shape: Seq[Int]

    def closure: ValueTerm

    // TODO: rename to enqueue
    def enqueue: Do[PendingBuffer]

  }

  trait CompiledKernel extends MonadicCloseable[UnitContinuation] {
    def run(parameters: List[Parameter]): Do[PendingBuffer]
  }

  protected def kernelCacheBuilder: CacheBuilder[ValueTerm, CompiledKernel] = {
    CacheBuilder
      .newBuilder()
      .removalListener(new RemovalListener[ValueTerm, CompiledKernel] {
        def onRemoval(notification: RemovalNotification[ValueTerm, CompiledKernel]): Unit = {
          val compiledKernel = notification.getValue
          compiledKernel.monadicClose.blockingAwait
        }
      })
  }

  protected val kernelCache: Cache[ValueTerm, CompiledKernel] = kernelCacheBuilder.build()

  protected implicit val executionContext: ExecutionContext

  private def clearCache: UnitContinuation[Unit] = UnitContinuation.execute {
    kernelCache.invalidateAll()
    kernelCache.cleanUp()
  }

  override def monadicClose: UnitContinuation[Unit] = {
    clearCache >> super.monadicClose
  }

  /** An intermediate expression of tensor that can be composed into a more complex expression.
    *
    * @note When this [[InlineTensor]] is referenced more than one expressions,
    *       the computation for the tensor may be evaluated more than once.
    * @see [[force]] to create a tensor that will cache the result.
    */
  trait InlineTensor extends Tensor {
    def force: BufferedTensor = {
      new {
//        val debuggingInformation: Implicitly[DebuggingInformation] = InlineTensor.this.debuggingInformation
        val shape: Seq[Int] = InlineTensor.this.shape
        val enqueue: Do[PendingBuffer] = InlineTensor.this.enqueue
      } with BufferedTensor
    }

    lazy val enqueue: Do[PendingBuffer] = {
      val compiledKernel = kernelCache.get(
        closure,
        new Callable[CompiledKernel] {
          def call(): CompiledKernel = {

            val alphConversionContext = new AlphaConversionContext
            val convertedTerm = closure.tree.alphaConversion(alphConversionContext).asInstanceOf[ValueTerm]

            val sourceCode = {
              val globalContext = new GlobalContext
              val functionContext = Factory[Context].newInstance(globalContext)

              val exportContext = new ExportContext
              val kernelBody = convertedTerm.tree.export(functionContext, exportContext)

              val kernelParameters = closure.upvalues.map { upvalue: Parameter =>
                exportContext.get(alphConversionContext.get(upvalue)).asInstanceOf[functionContext.Term]
              }
              fastraw"""
              ${globalContext.globalDeclarations}
              ${globalContext.globalDefinitions}
              ${functionContext.generateKernelSourceCode("kernel", shape.length, kernelParameters, Seq(kernelBody))}
              """
            }

            val program = createProgramWithSource(sourceCode)
            program.build()

            val compiledKernel = new CompiledKernel {

              def monadicClose: UnitContinuation[Unit] = program.monadicClose

              def run(upvalues: List[Parameter]): Do[PendingBuffer] = {
                // TODO: Manage life cycle of upvalues more delicately
                // e.g. a buffer should be release as soon as possible if it is a dependency of another buffer,
                // e.g. however, it can be hold longer time if it is dependencies of many other buffers.
                upvalues
                  .traverse { tree =>
                    tree.asInstanceOf[Parameter].id.asInstanceOf[Tensor].enqueue
                  }
                  .intransitiveFlatMap {
                    arguments: List[PendingBuffer] =>
                      Do.monadicCloseable(program.createFirstKernel()).intransitiveFlatMap { kernel: Kernel =>
                        allocateBuffer[Float](shape.product).flatMap { outputBuffer =>
                          for ((arugment, i) <- arguments.view.zipWithIndex) {
                            kernel(i) = arugment.buffer
                          }
                          kernel(arguments.length) = outputBuffer

                          kernel.enqueue(shape.view.map(_.toLong): _*).map { event0 =>
                            new PendingBuffer {
                              def event: Event = event0

                              def buffer: DeviceBuffer[Float] = outputBuffer
                            }

                          }

                        }
                      }
                  }

              }
            }
            kernelCache.put(convertedTerm, compiledKernel)
            compiledKernel
          }
        }
      )

      compiledKernel.run(closure.upvalues).shared
    }
  }

  trait TransformedTensor extends InlineTensor {

    def checkpoint: Tensor

    /** A matrix that describes the transformation of coordinate.
      *
      * The matrix size is __number of dimensions of original tensor × number of dimensions of new tensor__.
      */
    def matrix: RealMatrix

    // TODO: Add the transform operator in Expressions.scala
    val closure: ValueTerm = ???
  }

  trait BufferedTensor extends Tensor {
    val closure: ValueTerm = {
      array.parameter(this, float, shape: _*).extract
    }
  }

  def translate(previousTensor: Tensor, offset: Seq[Double]): Tensor = {
    translate(previousTensor, offset, previousTensor.shape)
  }

  def translate(previousTensor: Tensor,
                offset: Seq[Double],
                newShape: Seq[Int]) /*(implicit debuggingInformation0: Implicitly[DebuggingInformation])*/: Tensor = {
    if (offset.length != previousTensor.shape.length) {
      throw new IllegalArgumentException
    }

    previousTensor match {
      case previousTensor: TransformedTensor =>
        new TransformedTensor {
          val matrix: RealMatrix = {
            val newMatrix = previousTensor.matrix.copy()
            for (i <- offset.indices) {
              newMatrix.addToEntry(i, newMatrix.getColumnDimension - 1, offset(i))
            }
            newMatrix
          }
          val checkpoint: Tensor = previousTensor.checkpoint
          val shape: Seq[Int] = previousTensor.shape
//          val debuggingInformation: Implicitly[DebuggingInformation] = debuggingInformation0
        }
      case _ =>
        new TransformedTensor {
          val checkpoint: Tensor = previousTensor
          val shape: Seq[Int] = checkpoint.shape
//          val debuggingInformation: Implicitly[DebuggingInformation] = debuggingInformation0
          val matrix: RealMatrix = {
            val newMatrix = MatrixUtils.createRealMatrix(shape.length, shape.length + 1)
            for (i <- offset.indices) {
              newMatrix.setEntry(i, i, 1.0)
              newMatrix.setEntry(i, newMatrix.getColumnDimension - 1, offset(i))
            }
            newMatrix
          }
        }
    }
  }

}
