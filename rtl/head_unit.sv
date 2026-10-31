module head_unit #(
    parameter int GW = 12
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  clear,
    input  wire [255:0]          bias_flat,
    input  wire                  en,
    input  wire [2:0]            k,
    input  wire [GW-1:0]         g,
    input  wire signed [7:0]     w,
    output wire [255:0]          logits_flat,
    output wire [7:0]            mask
);

    reg signed [31:0] logit [0:7];
    wire signed [31:0] feature_value;
    wire signed [31:0] weight_value;
    wire signed [31:0] product_value;

    // Zero extension preserves the unsigned feature's nonnegative value.
    assign feature_value = $signed({{(32-GW){1'b0}}, g});
    assign weight_value = $signed({{24{w[7]}}, w});

    // The exact product fits in signed 32 bits for every supported GW.
    assign product_value = feature_value * weight_value;

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 8; i = i + 1) begin
                logit[i] <= 32'sd0;
            end
        end else if (clear) begin
            for (i = 0; i < 8; i = i + 1) begin
                logit[i] <= $signed(bias_flat[32*i +: 32]);
            end
        end else if (en) begin
            // A 32-bit addition implements modulo-2^32 accumulation.
            logit[k] <= logit[k] + product_value;
        end
    end

    genvar j;
    generate
        for (j = 0; j < 8; j = j + 1) begin : gen_outputs
            assign logits_flat[32*j +: 32] = logit[j];
            assign mask[j] = (logit[j] > 32'sd0);
        end
    endgenerate

endmodule
