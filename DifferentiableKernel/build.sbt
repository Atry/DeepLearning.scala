libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % Test

val lwjglNatives: String = {
  if (util.Properties.isMac) {
    "natives-macos"
  } else if (util.Properties.osName.startsWith("Linux")) {
    "natives-linux"
  } else if (util.Properties.isWin) {
    "natives-windows"
  } else {
    throw new MessageOnlyException(s"lwjgl does not support ${util.Properties.osName}")
  }
}

libraryDependencies += "org.lwjgl" % "lwjgl-opencl" % "3.1.1" % Test

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.1" % Test

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.1" % Test /* Runtime */ classifier lwjglNatives

fork := true

scalacOptions += "-Ypartial-unification"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "com.chrisneveu" %% "macrame" % "1.2.5"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3")
