function kronRz = kronRz(theta)
  % Theta is a vector of parameters to learn
  % Rz will have to be 1 at the bottom
  for k = 2:length(theta)
    if k == 2
      kronRz = kron(Rz(theta(1)),Rz(theta(2)));
    else
      kronRz = kron(kronRz, Rz(theta(k)));
    end
  end

  kronRz = kron(kronRz, eye(2));
end