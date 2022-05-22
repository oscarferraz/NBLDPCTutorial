 module bram(Clk,We,Waddr,Raddr,Din,Dout);
  input           Clk;
    input           We;
    input  [2:0]    Waddr;
    input  [2:0]    Raddr;
    input  [5:0]   Din;
    output [5:0]   Dout;
 

   (*ram_style="block"*) reg    [5:0]  mem[7:0]; 
    reg    [2:0]  raddr_reg;
 
    always @ (posedge Clk)
    begin
      raddr_reg  <= Raddr;
 
      if (We) begin     
        mem[Waddr]  <= Din;  
      end
 
      //Dout  <= mem[raddr_reg];   //registered read
 
    end
 
    assign Dout = mem[raddr_reg]; 
    endmodule