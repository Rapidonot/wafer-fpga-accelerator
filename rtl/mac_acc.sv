`default_nettype none

module mac_acc (
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    input  wire               in_first,
    input  wire               in_last,
    input  wire        [7:0]  act,
    input  wire signed [7:0]  wgt,
    input  wire signed [31:0] bias,
    output reg  signed [31:0] acc,
    output reg                out_valid
);

    wire signed [15:0] product;
    wire signed [31:0] product_extended;

    // Zero-extend the unsigned activation and sign-extend the weight.
    // Every possible product fits exactly in a signed 16-bit value.
    assign product = $signed({8'd0, act}) *
                     $signed({{8{wgt[7]}}, wgt});
    assign product_extended = {{16{product[15]}}, product};

    always @(posedge clk) begin
        if (rst) begin
            acc       <= 32'sd0;
            out_valid <= 1'b0;
        end else begin
            // The 32-bit addition wraps modulo 2**32.
            if (in_valid) begin
                acc <= (in_first ? bias : acc) + product_extended;
            end
            out_valid <= in_valid & in_last;
        end
    end

endmodule

`default_nettype wire
