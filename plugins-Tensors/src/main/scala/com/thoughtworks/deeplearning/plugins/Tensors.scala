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

import scala.language.existentials
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

  protected trait TermApi extends super.TermApi with ExpressionApi { this: Term =>
    def upvalues: Seq[Upvalue] = ???
    def alphaConversion(oldParameters: List[Term], renamedParameters: List[Term]): AlphaConversion[Self] = ???
  }

  type Term <: (Expression with Any) with TermApi

  protected trait UpvalueApi extends TermApi { this: Upvalue =>

    def bindUpvalue(kernel: Kernel, argumentIndex: Int): Unit

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

  type Upvalue <: (Term with Any) with UpvalueApi

  protected trait TensorUpvalueApi extends UpvalueApi { this: TensorUpvalue =>
    val tensor: Tensor
    def bindUpvalue(kernel: Kernel, argumentIndex: Int): Unit = ???
  }
  type TensorUpvalue <: (Upvalue with Any) with TensorUpvalueApi

  protected trait TypeApi extends super.TypeApi { this: Type =>
    @inject
    def TensorUpvalue: Operator1[Tensor, Identifier with TensorUpvalue]
//    @inject def Upvalue0: Operator0[Identifier with Upvalue]
  }

  /** @template */
  type Type <: (Expression with Any) with TypeApi

  // The scalar data type is hard-coded Float at the moment. FIXME: Allow other types in the future

  trait PendingBuffer {
    def event: Event
    def buffer: DeviceBuffer[Float]

    /** A matrix that describes the transformation of coordinate.
      *
      * The matrix size is ([[Tensor.shape]].length + 1) × [[Tensor.shape]].length
      */
    def matrix: RealMatrix
  }

  sealed trait Tensor { thisTensor =>
    def debuggingInformation: Implicitly[DebuggingInformation]

    def shape: Seq[Int]

    def kernelTerm: ValueTerm
    def computationalGraph: Do[PendingBuffer]

    def force: SharedTensor
  }

  trait Compiled extends MonadicCloseable[UnitContinuation] {
    def run(parameters: Seq[Upvalue]): Do[PendingBuffer]
  }

  protected def kernelCacheBuilder: CacheBuilder[StructuralKey, Compiled] = {
    CacheBuilder
      .newBuilder()
      .removalListener(new RemovalListener[StructuralKey, Compiled] {
        def onRemoval(notification: RemovalNotification[StructuralKey, Compiled]): Unit = {
          val kernelProgram = notification.getValue
          // TODO: clean up the kernel
          // kernelProgram.close()

        }
      })
  }

  protected val kernelCache: Cache[StructuralKey, Compiled] = kernelCacheBuilder.build()

  override def monadicClose: UnitContinuation[Unit] = {
    UnitContinuation.delay {
      kernelCache.invalidateAll()
      kernelCache.cleanUp()
    } >> super.monadicClose
  }

  /** An intermediate expression of tensor that can be composed into a more complex expression.
    *
    * @note When this [[InlineTensor]] is referenced more than one expressions,
    *       the computation for the tensor may be evaluated more than once.
    * @see [[force]] to create a tensor that will cache the result.
    */
  trait InlineTensor extends Tensor {
    def force: SharedTensor =
      new {
        val debuggingInformation: Implicitly[DebuggingInformation] = InlineTensor.this.debuggingInformation
        val shape: Seq[Int] = InlineTensor.this.shape
        val computationalGraph: Do[PendingBuffer] = InlineTensor.this.computationalGraph.shared
      } with SharedTensor

    // TODO: Cache the compiled OpenCL kernel
    // TODO: When comparing
    def computationalGraph: Do[PendingBuffer] = {
      val kernelProgram = kernelCache.get(
        new StructuralKey(kernelTerm),
        new Callable[Compiled] {
          def call(): Compiled = {

            val AlphaConversion(oldParameters, renamedParameters, convertedTerm) =
              kernelTerm.alphaConversion(Nil, Nil)
            val sourceCode = OpenCLValues.generateOpenCLKernelSourceCode(raw"""${kernelTerm.id}_kernel""",
                                                                         shape.length,
                                                                         renamedParameters,
                                                                         Seq(convertedTerm))
            val program = createProgramWithSource(sourceCode)
            program.build()

            new Compiled {

              def monadicClose: UnitContinuation[Unit] = program.monadicClose

              def run(upvalues: Seq[Upvalue]): Do[PendingBuffer] = {
                val kernel = program.createFirstKernel()
                for ((upvalue, i) <- upvalues.view.zipWithIndex) {
                  upvalue.bindUpvalue(kernel, i)
                }

                ???
              }
            }
          }
        }
      )

      kernelProgram.run(kernelTerm.upvalues)
    }
  }

  trait SharedTensor extends Tensor {

    val kernelTerm: ValueTerm = {
      val `type`: ArrayBufferType { type ElementType = float.type } =
        ArrayBufferType[float.type].newInstance(float, shape)
      val id: `type`.TypedTerm = `type`.TensorUpvalue(this)(debuggingInformation)
      id.extract(debuggingInformation)
    }

    def force: this.type = this

  }

  def translate(originalTensor: Tensor, offset: Seq[Double])(
      implicit debuggingInformation0: Implicitly[DebuggingInformation]): Tensor =
    new {
      val debuggingInformation = debuggingInformation0
      val shape: Seq[Int] = originalTensor.shape
    } with SharedTensor {
      def computationalGraph: Do[PendingBuffer] = {
        originalTensor.computationalGraph.map { originalPendingBuffer =>
          new PendingBuffer {
            def matrix: RealMatrix = {
              val matrixArray = (for (columnIndex <- (0 to shape.length).view) yield {
                (for (rowIndex <- (0 to shape.length).view) yield {
                  if (columnIndex == shape.length) {
                    originalPendingBuffer.matrix.getEntry(columnIndex, rowIndex) + offset(rowIndex)
                  } else {
                    originalPendingBuffer.matrix.getEntry(columnIndex, rowIndex)
                  }
                }).toArray
              }).toArray
              val copyArray = false
              new Array2DRowRealMatrix(matrixArray, copyArray)
            }

            val buffer: DeviceBuffer[Float] = originalPendingBuffer.buffer

            val event: Event = originalPendingBuffer.event
          }
        }
      }
    }

}
