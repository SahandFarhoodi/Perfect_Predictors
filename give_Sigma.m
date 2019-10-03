function [Sigma] = give_Sigma(dim, Sc_par)
    Sh_right = zeros(dim);
    for i = 1:size(Sh_right,1)-1
        Sh_right(i, i+1) = Sc_par;
    end
    Sh_left = Sh_right';
    Sigma = eye(dim);
    R = eye(dim);
    L = eye(dim);
    for i = 1:size(Sigma, 1)-1
        R = R*Sh_right;
        L = L*Sh_left;
        Sigma = Sigma + R + L; 
    end
end
