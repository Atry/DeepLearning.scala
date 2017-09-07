enablePlugins(Example)

exampleSuperTypes ~= { oldExampleSuperTypes =>
  import oldExampleSuperTypes._
  updated(indexOf("_root_.org.scalatest.FreeSpec"), "_root_.org.scalatest.AsyncFreeSpec")
}

exampleSuperTypes += "_root_.com.thoughtworks.deeplearning.scalatest.ThoughtworksFutureToScalaFuture"

libraryDependencies ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq("org.nd4j" % "nd4j-native-platform" % "0.9.1" % Test)
  }
}

fork in Test := true

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "org.plotly-scala" %% "plotly-render" % "0.3.1" % Test

libraryDependencies += "org.rauschig" % "jarchivelib" % "0.5.0" % Test

libraryDependencies += "com.thoughtworks.feature" %% "mixins-implicitssingleton" % "2.1.0-M0" % Test

scalacOptions += "-Ypartial-unification"
