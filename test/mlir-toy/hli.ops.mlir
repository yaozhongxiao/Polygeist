func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  %d = hli.add %a, %b : i32, i32 -> i32
  %f = hli.vadd %a, %b : i32, i32 -> i32
  %e = hli.sub %c, %d : i32, i32 -> i32
  %g = hli.const 32 : !hli.int<32>
  hli.return %e : i32
}