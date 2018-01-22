package com.thoughtworks.deeplearning.plugins

import java.util.concurrent.Callable

import com.google.common.cache._
import com.thoughtworks.compute.OpenCL
import com.thoughtworks.continuation._
import com.thoughtworks.expressions.Anonymous.Implicitly
import com.thoughtworks.expressions.Builtins.AllOpenCLExpressions
import com.thoughtworks.expressions.OpenCLValues
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

// TODO: Rename to VirtualTensors, like virtual-dom
trait Tensors extends OpenCL with AllOpenCLExpressions {

  final case class AlphaConversion[Root <: Term](oldParameters: List[Term],
                                                 renamedParameters: List[Term],
                                                 convertedTerm: Root)

  sealed trait StructuralComparison

  object StructuralComparison {

    final case class Same(thisParameters: List[Term], thatParameters: List[Term]) extends StructuralComparison

    case object Different extends StructuralComparison

  }

  final case class StructuralHashCode(thisParameters: List[Term], code: Int)

  // TODO: enable sbt-example for this plug-in.
  /** The key type for [[kernelCache]], which structurally performs [[hashCode]] and [[equals]] on [[term]].
    *
    * @example Given two [[Term]]s that contains the same operations but different [[Type#Identifier]],
    *
    *          {{{
    *          val a = float.Identifier()
    *          val b = float.Identifier()
    *          val x = float.Identifier()
    *          val y = float.Identifier()
    *          val aPlusB = a + b
    *          val xPlusY = x + y
    *          }}}
    *
    *          then they should not equal to each other,
    *
    *          {{{
    *          aPlusB shouldNot be(xPlusY)
    *          }}}
    *
    *          However, when wrappingThem as [[StructuralKey]],
    *
    *          {{{
    *          val structuralAPlusB = StructuralKey(aPlusB)
    *          val structuralXPlusY = StructuralKey(xPlusY)
    *          }}}
    *
    *          then they should equal to each other,
    *
    *          {{{
    *          structuralAPlusB should be(structuralXPlusY)
    *          }}}
    *
    *          and they should have the same [[hashCode]].
    *
    *          {{{
    *          structuralAPlusB.hashCode should be(structuralXPlusY.hashCode)
    *          }}}
    */
  final class StructuralKey(val term: Term) {

    override def hashCode(): Int = term.structuralHashCode(Nil).code

    override def equals(that: Any): Boolean = {
      that match {
        case that: StructuralKey =>
          term.isSameStructure(that.term)
        case _ =>
          false
      }
    }
  }

  protected trait ExpressionApi extends super.ExpressionApi {

    def isSameStructure(that: Term): Boolean = {
      structuralComparison(that, Nil, Nil) match {
        case _: StructuralComparison.Same =>
          true
        case StructuralComparison.Different =>
          false
      }
    }

    def structuralComparison(that: Term, thisParameters: List[Term], thatParameters: List[Term]): StructuralComparison =
      ???

    def structuralHashCode(thisParameters: List[Term]): StructuralHashCode = ???

  }

  type Expression <: ExpressionApi

  protected trait TermApi extends super.TermApi with ExpressionApi {
    this: Term =>
    def upvalues: List[TensorUpvalue] = ???

    def alphaConversion(oldParameters: List[Term], renamedParameters: List[Term]): AlphaConversion[Self] = ???
  }

  type Term <: (Expression with Any) with TermApi

  protected trait TensorUpvalueApi extends TermApi {
    this: TensorUpvalue =>
    val tensor: Tensor

    override def alphaConversion(oldParameters: List[Term], renamedParameters: List[Term]): AlphaConversion[Self] = {
      oldParameters.indexOf(this) match {
        case -1 =>
          val renamedParameter = `type`.Identifier()(debuggingInformation)
          AlphaConversion[Self](this :: oldParameters, renamedParameter :: renamedParameters, renamedParameter)
        case index =>
          val renamedParameter = renamedParameters(index).asInstanceOf[Self]
          AlphaConversion[Self](oldParameters, renamedParameters, renamedParameter)
      }
    }
  }

  type TensorUpvalue <: (Term with Any) with TensorUpvalueApi

  protected trait TypeApi extends super.TypeApi {
    this: Type =>
    @inject
    def TensorUpvalue: Operator1[Tensor, Identifier with TensorUpvalue]
  }

  /** @template */
  type Type <: (Expression with Any) with TypeApi

  // The scalar data type is hard-coded Float at the moment. FIXME: Allow other types in the future
  trait PendingBuffer {
    def event: Event

    def buffer: DeviceBuffer[Float]
  }

  sealed trait Tensor {
    thisTensor =>
    def debuggingInformation: Implicitly[DebuggingInformation]

    def shape: Seq[Int]

    def kernelTerm: ValueTerm

    def fill: Do[PendingBuffer]

  }

  trait CompiledKernel extends MonadicCloseable[UnitContinuation] {
    def run(parameters: List[TensorUpvalue]): Do[PendingBuffer]
  }

  protected def kernelCacheBuilder: CacheBuilder[StructuralKey, CompiledKernel] = {
    CacheBuilder
      .newBuilder()
      .removalListener(new RemovalListener[StructuralKey, CompiledKernel] {
        def onRemoval(notification: RemovalNotification[StructuralKey, CompiledKernel]): Unit = {
          val compiledKernel = notification.getValue
          compiledKernel.monadicClose.reset
        }
      })
  }

  protected val kernelCache: Cache[StructuralKey, CompiledKernel] = kernelCacheBuilder.build()

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
        val debuggingInformation: Implicitly[DebuggingInformation] = InlineTensor.this.debuggingInformation
        val shape: Seq[Int] = InlineTensor.this.shape
        val fill: Do[PendingBuffer] = InlineTensor.this.fill
      } with BufferedTensor
    }

    lazy val fill: Do[PendingBuffer] = {
      val compiledKernel = kernelCache.get(
        new StructuralKey(kernelTerm),
        new Callable[CompiledKernel] {
          def call(): CompiledKernel = {

            val AlphaConversion(oldParameters, renamedParameters, convertedTerm) =
              kernelTerm.alphaConversion(Nil, Nil)
            val sourceCode = OpenCLValues.generateOpenCLKernelSourceCode(raw"""${kernelTerm.id}_kernel""",
                                                                         shape.length,
                                                                         renamedParameters,
                                                                         Seq(convertedTerm))
            val program = createProgramWithSource(sourceCode)
            program.build()

            new CompiledKernel {

              def monadicClose: UnitContinuation[Unit] = program.monadicClose

              def run(upvalues: List[TensorUpvalue]): Do[PendingBuffer] = {
                // TODO: Manage life cycle of upvalues more delicately
                // e.g. a buffer should be release as soon as possible if it is a dependency of another buffer,
                // e.g. however, it can be hold longer time if it is dependencies of many other buffers.
                upvalues.traverse(_.tensor.fill).intransitiveFlatMap {
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
          }
        }
      )

      compiledKernel.run(kernelTerm.upvalues).shared
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
    val kernelTerm: ValueTerm = ???
  }

  trait BufferedTensor extends Tensor {

    val kernelTerm: ValueTerm = {
      val `type`: ArrayBufferType { type ElementType = float.type } =
        ArrayBufferType[float.type].newInstance(float, shape)
      val id: `type`.TypedTerm = `type`.TensorUpvalue(this)(debuggingInformation)
      id.extract(debuggingInformation)
    }

  }

  def translate(previousTensor: Tensor, offset: Seq[Double])(
      implicit debuggingInformation0: Implicitly[DebuggingInformation]): Tensor = {
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
          val debuggingInformation: Implicitly[DebuggingInformation] = debuggingInformation0
        }
      case _ =>
        new TransformedTensor {
          val checkpoint: Tensor = previousTensor
          val shape: Seq[Int] = checkpoint.shape
          val debuggingInformation: Implicitly[DebuggingInformation] = debuggingInformation0
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
