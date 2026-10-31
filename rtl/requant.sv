`default_nettype none

module requant (
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    input  wire signed [31:0] acc,
    input  wire        [15:0] m,
    input  wire        [5:0]  s,
    output reg                out_valid,
    output reg         [7:0]  q
);

    reg signed [48:0] product_stage1;
    reg        [5:0]  shift_stage1;
    reg               valid_stage1;

    wire signed [48:0] product_next;
    wire signed [48:0] rounding;
    wire signed [48:0] rounded_product;
    wire signed [48:0] shifted_value;
    wire        [7:0]  saturated_value;

    // The assignment context provides 49-bit signed multiplication.
    // Adding a zero sign bit makes the unsigned multiplier nonnegative.
    assign product_next = $signed(acc) * $signed({1'b0, m});

    // All arithmetic stays signed and 49 bits wide. For legal inputs,
    // the exact product plus rounding constant fits within this width.
    assign rounding = 49'sd1 <<< (shift_stage1 - 6'd1);
    assign rounded_product = product_stage1 + rounding;
    assign shifted_value = rounded_product >>> shift_stage1;

    assign saturated_value = (shifted_value < 49'sd0)   ? 8'd0 :
                             (shifted_value > 49'sd255) ? 8'd255 :
                                                         shifted_value[7:0];

    always @(posedge clk) begin
        if (rst) begin
            product_stage1 <= 49'sd0;
            shift_stage1   <= 6'd1;
            valid_stage1   <= 1'b0;
            out_valid      <= 1'b0;
            q              <= 8'd0;
        end else begin
            // Stage 1: capture the exact product and transaction metadata.
            product_stage1 <= product_next;
            shift_stage1   <= s;
            valid_stage1   <= in_valid;

            // Stage 2: publish the preceding stage-1 transaction.
            out_valid <= valid_stage1;
            if (valid_stage1) begin
                q <= saturated_value;
            end
        end
    end

endmodule

`default_nettype wire
