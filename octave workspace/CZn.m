function CZn = CZn(n)
% gate is any 4x4 entangling gate like CNOT/CX or CZ
% n is the number of wires we are going to use
% We will entangle all the gates, cascading entangled gate.

if n < 2
  display("ERROR: n has to be greater than 2");
  return
elseif n>13
  display("ERROR: n greater than 13 will lag your computer");
  return
end

% Key Variables
Z = [1 0;0 -1];     % Pauli Z
d0 = [1;0]*[1 0];   % density matrix of |0><0|
d1 = [0;1]*[0 1];   % density matrix of |1><1|


CZn = kron(d0, eye(2^(n-1)));

if n == 2
  CZn += kron(d1,Z);
  
elseif n == 3
  CZn += kron(d1,d0,eye(2));
  CZn += kron(d1,d1,Z);
    
elseif n == 4
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,Z);

elseif n == 5
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,Z);

elseif n == 6
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,Z);
  
elseif n == 7
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,Z);
  
elseif n == 8
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d0,eye(2^(n-7)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,Z);  
  
elseif n == 9
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d0,eye(2^(n-7)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-8)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,Z);  
  
elseif n == 10
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d0,eye(2^(n-7)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-8)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-9))); 
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,Z); 

elseif n == 11
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d0,eye(2^(n-7)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-8)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-9))); 
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-10)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,Z);
  
elseif n == 12
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d0,eye(2^(n-7)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-8)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-9))); 
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-10)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-11)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,Z);
  
elseif n == 13
  CZn += kron(d1,d0,eye(2^(n-2)));
  CZn += kron(d1,d1,d0,eye(2^(n-3)));
  CZn += kron(d1,d1,d1,d0,eye(2^(n-4)));
  CZn += kron(d1,d1,d1,d1,d0,eye(2^(n-5)));
  CZn += kron(d1,d1,d1,d1,d1,d0,eye(2^(n-6)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d0,eye(2^(n-7)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-8)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-9))); 
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-10)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-11)));
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d0,eye(2^(n-12)));  
  CZn += kron(d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,d1,Z);
    
end   
 
end

