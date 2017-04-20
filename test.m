
clear all, close all, clc
Bim = imread('derek4', 'jpeg');
B = double(Bim);
[nx,ny,nz] = size(B); %nz = 3 for RGB and 1 for BW
x=linspace(0,1,nx); y=linspace(0,1,ny); dx=x(2)-x(1); dy=y(2)-y(1);
onex=ones(nx,1); oney=ones(ny,1);
Dx=(spdiags([onex -2*onex onex],[-1 0 1],nx,nx))/dx^2; Ix=eye(nx);
Dy=(spdiags([oney -2*oney oney],[-1 0 1],ny,ny))/dy^2; Iy=eye(ny);
L=kron(Iy,Dx)+kron(Dy,Ix);
%kron function converts operator to higher dimensional

%space, similar to the use of meshgrid.
%This part construct the variable diffusion constant
Dvar = zeros(size(B));
spotcx = floor(nx*19/32);
spotcy = floor(ny*59/128);
width = 15; % width of area to which we have applied diffusion
dConst = 0.001; %strength of diffusion constant when applied.
Dvar((spotcx-width):1:(spotcx+width),(spotcy-width):1:(spotcy+width)) ...
=dConst*ones(2*width+1,2*width+1);
Dk = reshape(Dvar, nx*ny, 1); % reshape for proper use in ode function
%Performs actual diffusion computation
tspan=[0 0.06:0.02:0.1]; u=reshape(B,nx*ny,1);
[t,usol]=ode113('image_rhs',tspan,u,[],L,Dk);
%Here we have examined several diffusion times (tspan)
for j=1:length(t)
A_clean=uint8(reshape(usol(j,:),nx,ny));
A_clean(spotcx, spotcy) = 255;
subplot(2,ceil(length(t)/2), j), imshow(A_clean);
title(['Diffusion Time: ', num2str(t(j))]);
end








