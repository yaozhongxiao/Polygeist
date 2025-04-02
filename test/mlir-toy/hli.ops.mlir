func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  %d = hli.add %a, %b : i32, i32 -> i32
  %e = hli.sub %c, %d : i32, i32 -> i32
  func.return %e : i32
}