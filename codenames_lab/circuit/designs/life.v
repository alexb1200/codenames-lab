// Game of Life on a W x H torus, unrolled for T generations.
// Purely combinational: s0 is generation 0, hist holds generations 1..T.
// Bit (g-1)*W*H + y*W + x of hist is cell (x, y) in generation g.

module life_cell(input self, input [7:0] nb, output next);
  wire [3:0] count = nb[0] + nb[1] + nb[2] + nb[3]
                   + nb[4] + nb[5] + nb[6] + nb[7];
  assign next = (count == 4'd3) | (self & (count == 4'd2));
endmodule

module life_step #(parameter W = 6, H = 6)
                  (input [W*H-1:0] s, output [W*H-1:0] t);
  genvar x, y;
  generate
    for (y = 0; y < H; y = y + 1) begin : row
      for (x = 0; x < W; x = x + 1) begin : col
        localparam XL = (x + W - 1) % W, XR = (x + 1) % W;
        localparam YU = (y + H - 1) % H, YD = (y + 1) % H;
        life_cell c(
          .self(s[y*W + x]),
          .nb({s[YU*W + XL], s[YU*W + x], s[YU*W + XR],
               s[y*W + XL],                s[y*W + XR],
               s[YD*W + XL], s[YD*W + x], s[YD*W + XR]}),
          .next(t[y*W + x]));
      end
    end
  endgenerate
endmodule

module life #(parameter W = 6, H = 6, T = 4)
             (input [W*H-1:0] s0, output [T*W*H-1:0] hist);
  wire [W*H-1:0] g [0:T];
  assign g[0] = s0;
  genvar k;
  generate
    for (k = 0; k < T; k = k + 1) begin : gen
      life_step #(.W(W), .H(H)) st(.s(g[k]), .t(g[k+1]));
      assign hist[k*W*H +: W*H] = g[k+1];
    end
  endgenerate
endmodule
