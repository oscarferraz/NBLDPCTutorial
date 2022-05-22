`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11.11.2020 12:24:43
// Design Name: 
// Module Name: blocking_counter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module counter_even(reset,clk,enable,count);
input reset,clk;
input enable;
output reg[9:0]count;

always@(posedge clk or posedge reset)
begin
if(reset) begin

count=10'd0;

end
else if(enable)
begin
count=count+2;
end
else begin
count=count;
end

end
endmodule
