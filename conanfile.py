from conans import ConanFile

class GrppiConan(ConanFile):
    name = "GrPPI"
    version = "0.4.0"
    license = "Apache License - Version 2.0 - January 2004"
    url = "https://github.com/arcosuc3m/grppi"
    description = """GrPPI is an open source generic and reusable parallel pattern programming interface developed at University Carlos III of Madrid"""
    exports_sources = "include/*", "LICENSE"
    build_policy = "missing"

    def package(self):
        self.copy("*.h", src="include", dst="include", keep_path=True)
        self.copy("LICENSE", dst="licenses", ignore_case=True, keep_path=False)

