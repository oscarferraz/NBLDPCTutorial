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


module counter_disable_b(reset,clk,enable,count,enable0);
input reset,clk;
input enable;
input enable0;
output reg[9:0]count;

always@(posedge clk or posedge reset)
begin
if(reset) begin

count=10'd0;

end
else if(enable)
begin
if(enable0==0)
count=10'b1_1111_1111_0;
else if(enable0==1)
count=count+2;
end
else begin
count=count;
end

end

endmodule
