import sbt.Keys._
import sbt.plugins.JvmPlugin
import sbt._
import scala.language.experimental.macros

/**
  * @author 杨博 (Yang Bo)
  */
object ExportClasspath extends AutoPlugin {

  object autoImport {
    val exportClasspath =
      taskKey[File]("Export the full classpath to a sc file, which can be loaded from Ammonite REPL or Jupyter Scala.")
  }
  import autoImport._

  override def trigger: PluginTrigger = allRequirements

  override def requires: Plugins = JvmPlugin

  override def projectSettings: Seq[Def.Setting[_]] =
    Seq(Test, Compile, Runtime).flatMap { configuration =>
      inConfig(configuration)(
        Seq(
          exportClasspath := {
            import scala.reflect.runtime.universe._
            def pathTrees = fullClasspath.value.map { attributedFile =>
              q"_root_.ammonite.ops.Path(${attributedFile.data.toString})"
            }
            def loadTree = q"interp.load.cp(Seq(..$pathTrees))"
            val scFile = crossTarget.value / raw"""classpath-${configuration.name}.sc"""
            IO.write(scFile, show(loadTree))
            scFile
          }
        ))
    }

}
