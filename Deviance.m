function out = Deviance(Y, X, b)
    l = exp(X*b);
    z = find(Y>0);
    Y = repmat(Y, 1, size(b, 2));
    m = -sum(l)+sum(Y(z).*log(l(z)));
    out = 2*(-sum(Y) + sum(Y(z).*log(Y(z))) - m);
    out = out';
end