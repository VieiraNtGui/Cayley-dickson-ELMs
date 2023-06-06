## AUTOENCODING TASK ACCORDING TO MINEMOTO ET AL. 2017
## RESULTS ARE PART OF THE PAPER "EXTREME LEARNING MACHINES ON CAYLEY-DICKSON ALGEBRAS APPLIED FOR COLOR IMAGE AUTO-ENCODING", VIEIRA AND VALLE, 2020.
## PERFORMANCE COMPARISON FOR REAL VS CAYLEY-DICKSON NETWORKS

include("Prod_Cayley.jl")
using StatsFuns
using LinearAlgebra
using Plots
using ODE
using Quaternions
using Statistics
using Distributions
using Random
using CSV, DataFrames
using Plots
using MAT
using TickTock
using ImageView, Images


batch1 = matread("..\\CIFAR-10\\data_batch_1.mat")
test = matread("..\\CIFAR-10\\test_batch.mat")


A = convert(Array{Float64,2},batch1["data"])
A = 2*(A[1:10,:]./255) .-1
A_cd = cat(zeros(size(A)[1],1024),A[:,1:1024],A[:,1025:2048],A[:,2049:3072],dims=3)
T = convert(Array{Float64,2},test["data"])
T = 2*(T[1:10,:]./255) .-1
T_cd = cat(zeros(size(T)[1],1024),T[:,1:1024],T[:,1025:2048],T[:,2049:3072],dims=3)


# AUTOENCODING WITH REAL PARAMETERS
tick()
lr = 600
W = randn(3072,lr)

H = tanh.(A*W)
M = H\A
Y_r = H*M
diff = (H*M - A).^2
diff = mean(diff, dims=2)
RMSE_R = mean(sqrt,diff)

H_test = tanh.(T*W)
Y_test = H_test*M
diff_test = (H_test*M - T).^2
diff_test = mean(diff_test, dims=2)
RMSE_R_test = mean(sqrt,diff_test)
tock()

histogram(vcat(H_cd...))
histogram(vcat(H_test...))

imshow(reshape(A[1,:],32,32,3))
imshow(reshape(Y_r[1,:],32,32,3))
imshow(reshape(T[6,:],32,32,3))
imshow(reshape(Y_test[6,:],32,32,3))


# AUTOENCODING IN CAYLEY-DICKSON ALGEBRA
gamma = [+1,+1]
K = convert(Int64,size(gamma)[1])
l_q = 449 # number of hidden neurons

alpha=0.4
u = Uniform(alpha*-1,alpha*1)
W_cd = rand(u,size(A_cd)[2],l_q,2^K)
H_cd = zeros(size(A_cd)[1],l_q,2^K)
B = zeros(2^K * size(A_cd)[1],2^K * l_q)
tick()
for i = 1:size(A_cd)[1]
    println(i)
    for j = 1:l_q
        for k = 1:size(A_cd)[2]
            H_cd[i,j,:] += prodCayley(T_cd[i,k,:],W_cd[k,j,:],gamma)
        end
    end
end
tock()
H_cd = tanh.(H_cd)
for i = 1:size(A_cd)[1]
    for k = 1:l_q
        B[4*(i-1)+1:4*i,4*(k-1)+1:4*k] = input2ff(reshape(H_cd[i,k,:],1,1,4))
    end
end
K = convert(Int64,size(gamma)[1])
desired = zeros(2^K * size(A_cd)[1],size(A_cd)[2])
for i=1:size(A_cd)[1]
    for j = 1:2^K
        desired[2^K*(i-1)+j,:] = A_cd[i,:,j]
    end
end
M_cd = B\desired
out = B*M_cd
answer = zeros(size(A_cd))
for i=1:size(A_cd)[1]
    for j=1:2^K
        answer[i,:,j] = out[2^K * (i-1)+j,:]
    end
end
diff_cd = (A_cd-answer).^2
diff_cd = sum(diff_cd,dims=3)
diff_cd = (1/(4*size(A_cd)[2]))*mean(diff_cd,dims=2)
RMSE_cd = mean(sqrt,diff_cd)
println("Training error is: ", string(RMSE_cd))

answer, W_cd, M_cd = elm_cd(A_cd,A_cd,gamma,449)


# TEST SET
H_T_cd = zeros(size(T_cd)[1],l_q,2^K)
for i = 1:size(T_cd)[1]
    for j = 1:l_q
        for k = 1:size(T_cd)[2]
            H_T_cd[i,j,:] += prodCayley(T_cd[i,k,:],W_cd[k,j,:],gamma)
        end
    end
end
H_T_cd = tanh.(H_T_cd)
B_T = zeros(2^K * size(T_cd)[1],2^K * l_q)
for i = 1:size(T_cd)[1]
    for k = 1:l_q
        B_T[4*(i-1)+1:4*i,4*(k-1)+1:4*k] = input2ff(reshape(H_T_cd[i,k,:],1,1,4))
    end
end
out_T = B_T*M_cd
answer_T = zeros(size(T_cd))
for i=1:size(T_cd)[1]
    for j=1:2^K
        answer_T[i,:,j] = out_T[2^K * (i-1)+j,:]
    end
end

diff_cd_T = (T_cd-answer_T).^2
diff_cd_T = sum(diff_cd_T,dims=3)
diff_cd_T = (1/(4*size(T_cd)[2]))*mean(diff_cd_T,dims=2)
RMSE_cd_T = mean(sqrt,diff_cd_T)

ff_test(T_cd,W_cd,M_cd,T_cd,gamma)

imshow(reshape(A_cd[1,:,2:4],32,32,3))
imshow(reshape(answer[5,:,2:4],32,32,3))
for i=1:10
    imshow(reshape(T_cd[i,:,2:4],32,32,3))
end
for i=1:10
    imshow(reshape(answer_T[i,:,2:4],32,32,3))
end
# matwrite("train_output_batch2.mat", Dict("answer" => collect(answer)
# ); compress = true)
# matwrite("hidden_weights_batch2.mat", Dict("W_cd" => collect(H_cd)
# ); compress = true)
# matwrite("output_weights_batch2.mat", Dict("M_cd" => collect(M_cd)
# ); compress = true)
