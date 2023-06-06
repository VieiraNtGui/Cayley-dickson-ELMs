# Cayley-Dickson algebra algebra

function MProdCayley(x,y,gamma)
    # Select the less expensive way to compute the product.
    if length(x)<length(y)
        X = left2matrix(x,gamma)
        Y = Cayley2matrix(y)
        Z = X*Y
        return matrix2Cayley(Z,gamma)
    else
        X = RCayley2matrix(x)
        Y = right2matrix(y,gamma)
        Z = Y*X
        return Rmatrix2Cayley(Z,gamma)
    end
end

function prodCayley(x,y,gamma)       # Computes x.y in the algebra with spanning param. gamma

    K = convert(Int64,length(gamma))
    if K>0
        N = 2^K;
        M = convert(Int64,N / 2)

        a = x[1:M]
        b = x[M+1:N]
        c = y[1:M]
        d = y[M+1:N]

        gamma_ant = gamma[1:K-1]
        z = [prodCayley(a,c,gamma_ant)+gamma[K]*prodCayley(conjCayley(d),b,gamma_ant); prodCayley(d,a,gamma_ant)+prodCayley(b,conjCayley(c),gamma_ant)];
    else
        z = x.*y
    end
    return z
end

function conjCayley(x)             # classic involution
    N = length(x)
    K = convert(Int64,log2(N))

    if K>0
        M = convert(Int64,N/2)
        a = x[1:M]
        b = x[M+1:N]
        z = [conjCayley(a); -b];
    else
        z = x
    end
    return z
end

e(i,k) = [zeros(i-1,1);1;zeros(2^k-i,1)]     # generates the i-th canonic vector of the basis of the dim 2^k space

function l2matrix(x,gamma)    # Transforms vector x of the algebra w/ param. gamma into the product-by-left associated matrix
    N = length(x)                 # i.e. Ax => XA.
    K = convert(Int64, log2(N))
    A = zeros(N,N)
    for i=1:N
        A[:,i] = prodCayley(x,e(i,K),gamma)
    end
    return A
end        

function left2matrix(x,gamma)               # same as above but for matrices
    M = size(x)[1]                
    N = size(x)[2]
    Kx = size(x)[3]


    X = rand(Kx*M,Kx*N)
    ind = [i for i=1:Kx]
    for i=1:M
        for j=1:N
            X[Kx*(i-1).+ind,Kx*(j-1).+ind] = l2matrix(x[i,j,:],gamma)
        end
    end
    return X
end

function r2matrix(x,gamma)    # Transforms vector x of the algebra w/ param. gamma into the product-by-right associated matrix
    N = length(x)                 # i.e. Ax => XA.
    K = convert(Int64, log2(N))
    A = zeros(N,N)
    for i=1:N
        A[:,i] = prodCayley(e(i,K),x,gamma)
    end
    return A
end        

function right2matrix(x,gamma)              # same as above but for matrices
    M = size(x)[1]                
    N = size(x)[2]
    Kx = size(x)[3]

    X = zeros(Kx*N,Kx*M)
    ind = [i for i=1:Kx]
    for i=1:M
        for j=1:N
            X[Kx*(j-1).+ind,Kx*(i-1).+ind] = r2matrix(x[i,j,:],gamma)
        end
    end
    return X
end
        
function Cayley2matrix(x)    
    M = size(x)[1]                
    N = size(x)[2]
    Kx = size(x)[3]
    return reshape(permutedims(x,[3,1,2]),Kx*M,N)
end
   
function RCayley2matrix(x)
   return Cayley2matrix(permutedims(x,[2,1,3]))
end

function matrix2Cayley(X,gamma)
    K = length(gamma)
    Kx = 2^K
    M = convert(Int64,size(X,1)/Kx)
    N = size(X,2)
    ind = [i for i=1:4]
    x = zeros(M,N,4)
    for i=1:M
        for j=1:N
            x[i,j,:] = X[Kx*(i-1).+ind,j]
        end
    end
    return x
end

function Rmatrix2Cayley(X,gamma)
    return permutedims(matrix2Cayley(X,gamma),[2,1,3])
end
        
function TrainCayleyELM(X,Y,N_hidden,gamma)
    # Trains the Cayley-Dickson ELM
    K = length(gamma)
    alpha = 10/size(X)[2]
    W = alpha*randn(size(X)[2],N_hidden,2^K);
    H = tanh.(MProdCayley(X,W,gamma));
    Hm = left2matrix(H,gamma);
    Zm = Cayley2matrix(Y);
    M = matrix2Cayley(Hm\Zm,gamma);
    return W, M
end

function EvalCayleyELM(W,M,X,gamma)
    H = tanh.(MProdCayley(X,W,gamma))
    return MProdCayley(H,M,gamma)
end
        
function TrainELM(X,Y,N_hidden)
    # Trains the real-valued ELM
    alpha = 30/size(X)[2]
    W = alpha*randn(size(X)[2],N_hidden);
    H = tanh.(X*W);
    M = H\Y;
    return W, M
end
        
function EvalELM(W,M,X)
    H = tanh.(X*W)
    return H*M
end

function MED(x,y,dim=2)  # mean euclidean distance
    diff = ((x-y).^2)
    diff = sum(diff,dims=dim).^(1/2)
    ed = mean(diff,dims=1)
    sd = std(diff,dims=1)
    return ed[1], sd[1]
end
        