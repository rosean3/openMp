{ pkgs }: {
	deps = [
   pkgs.gnuplot
		pkgs.clang
		pkgs.ccls
		pkgs.gdb
		pkgs.gnumake
	];
}