// 4-bit adder with carry out: a second design, to show the compiler is general.
module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
  assign sum = a + b;
endmodule
