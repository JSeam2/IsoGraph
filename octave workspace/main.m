% Constants
% Qubit
s1 = [0;1];
s0 = [1;0];

% Gates
% Hadamard
H = [ 1  1;
      1 -1 ]/sqrt(2); 

        
% sanity check 3x3 adjacency
% 0 1 0     0 0 1     
% 1 0 1     0 0 1
% 0 1 0     1 1 0

% Upper Diagonal as inputs, final qubit as output
% Input state |1 0 1 0 1 1 0>

input = kron(s1,s0,s1,s0,s1,s1,s0);
kronH = kron(H,H,H,H,H,H,eye(2));
CZ7 = CZn(7);

% Learn theta parameters 1
theta1 = rand(6,1);
kronRz1 = kronRz(theta1);
block1 = CZ7* kronRz1* kronH;

% Learn theta parameters 2
theta2 = rand(6,1);
kronRz2 = kronRz(theta2);
block2 = CZ7* kronRz2* kronH;

% Learn theta parameters 3
theta3 = rand(6,1);
kronRz3 = kronRz(theta3);
block3 = CZ7* kronRz3* kronH;

% Note you multiply in the opposite direction
final_state = block3*block2*block1*input;

% get density matrix
rho = final_state*final_state';
%trace(rho) == 1

% projectors
p0 = kron(eye(2^6), s0*s0');
p1 = kron(eye(2^6), s1*s1');

% Sum of projectors
%pp = p0*rho*p0' + p1*rho*p1'

prob_0 = trace(p0'*p0*rho)
prob_1 = trace(p1'*p1*rho)

