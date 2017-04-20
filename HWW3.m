clear all; close all; clc
% %% Part one - Black and white
% % Black and white image with filter
% figure(1)
% An = imread('derek2','jpeg');
% subplot(1,2,1)
% 
% imshow(An)
% title('Original Image')
% [nx,ny] = size(An);
% 
% % FFT the image
% Ant = fft2(An);
% figure(1)
% subplot(1,2,2)
% pcolor(log(abs(fftshift(Ant)) + 1)), shading interp
% title('Fourier frequency(Log)')
% 
% % Turns out the highest frequency at the center of the fourier domain
% % Where to apply a gaussian filter
% xcent = nx/2;
% ycent = ny/2;
% 
% % Create a gaussian filter with certain width
% kx = 1 : nx; ky = 1 : ny;
% [Kx,Ky] = meshgrid(kx,ky);
% figure(2)
% subplot(3,3,1)
% imshow(An)
% title('Original Image')
% sigma = 100;
% for j = 2 :9
% sigma = sigma/10;
% F = exp(-sigma*(Kx - xcent).^2 - sigma*(Ky - ycent).^2);
% % Shift the filter to fit the fft domain
% Fs = fftshift(F);
% Atf = Ant.* Fs';
% 
% % Transform back
% Af = ifft2(Atf);
% A = uint8(Af);
% subplot(3,3,j)
% imshow(A);
% title(['sigma =' num2str(sigma)]);
% end

% %% Part two - Color image
% AA = imread('derek1','jpeg');
% An = double(AA);
% % take out each layer of the data cube
% Rn = An(:,:,1);
% Gn = An(:,:,2);
% Bn = An(:,:,3);
% [nx,ny] = size(Bn);
% % Do fft to find where to apply filter
% % for R:
% figure(1)
% Rnt = fft2(Rn);
% subplot(2,2,1)
% pcolor(log(abs(fftshift(Rnt)) + 1)), shading interp
% title('Fourier frequency(Log) for R')
% 
% Gnt = fft2(Gn);
% subplot(2,2,2)
% pcolor(log(abs(fftshift(Gnt)) + 1)), shading interp
% title('Fourier frequency(Log) for G')
% 
% Bnt = fft2(Bn);
% subplot(2,2,3)
% pcolor(log(abs(fftshift(Bnt)) + 1)), shading interp
% title('Fourier frequency(Log) for B')
% 
% % Turns out still at the center of fourier domain
% % Apply filter 
% xcent = nx/2;
% ycent = ny/2;
% kx = 1 : nx; 
% ky = 1 : ny;
% [Kx,Ky] = meshgrid(kx,ky);
% 
% % Apply different filter width 
% sigma = [100 10 1 0.1 0.01 0.001 0.0001 0.00001 0.000001]
% figure(2)
% subplot(3,3,1)
% imshow(AA)
% title('Original Image')
% for ii = 2:length(sigma)
% F = exp(-sigma(ii)*(Kx - xcent).^2 - sigma(ii)*(Ky - ycent).^2); 
% % convert the filter in to fourier domain
% Fs = fftshift(F);
% A = zeros(nx,ny,3);
% % Filter each layer for RGB
% for j = 1:3
%     Ant = fft2(An(:,:,j));
%     Ants = Ant;
%     Antsf = Ants.*Fs';
%     A(:,:,j) = ifft2(Antsf);
% end
% subplot(3,3,ii)
% imshow(uint8(A));
% title(['sigma =' num2str(sigma(ii))]);
% end

% %% Part two - Black and white
% clear all, close all, clc
% A = imread('derek4', 'jpeg');
% 
% A = double(A);
% [nx,ny] = size(A);
% x = linspace(0,1,nx); dx = x(2) - x(1);
% y = linspace(0,1,ny); dy = y(2) - y(1);
% 
% onex = ones(nx,1); oney = ones(ny,1);
% % track only non-zero entries
% % second derivative in x direction
% Dx = (spdiags([onex -2*onex onex],[-1 0 1], nx, nx)/dx^2);
% Ix = eye(nx);
% % in y direction
% Dy = (spdiags([oney -2*oney oney],[-1 0 1], ny, ny)/dy^2);
% Iy = eye(ny);
% 
% % make the meshgrid for derivative
% L=kron(Iy,Dx)+kron(Dy,Ix);
% % make D with xy domain
% % By playing around find that the rash is on 
% % A(135:165,155:185)
% D = zeros(size(A));
% for j = 135:165
%     for jj = 155:185
%         D(j,jj) = 1;
%     end
% end
% 
% An0 = reshape(A,nx*ny,1);
% tspan = [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08];
% Dc = 0.001;
% D = Dc*D;
% Ds = reshape(D, nx*ny, 1);
% [t,A_sol] = ode45('image_rhs',tspan, An0,[],L,Ds);
% 
% % pull the vector out in to matrix
% figure(1)
% for j = 1:9
%     subplot(3,3,j)
%     Aclean = uint8(reshape(A_sol(j,:),nx,ny));
%     imshow(Aclean);
%     title(['Time: ', num2str(tspan(j))]);
% end

%% Part two - diffusion with RGB
clear all;close all; clc
A = imread('derek3', 'jpeg');
A = double(A);
[nx,ny,nz] = size(A);
x = linspace(0,1,nx); dx = x(2) - x(1);
y = linspace(0,1,ny); dy = y(2) - y(1);

onex = ones(nx,1); oney = ones(ny,1);
% track only non-zero entries
% second derivative in x direction
Dx = (spdiags([onex -2*onex onex],[-1 0 1], nx, nx)/dx^2);
Ix = eye(nx);
% in y direction
Dy = (spdiags([oney -2*oney oney],[-1 0 1], ny, ny)/dy^2);
Iy = eye(ny);

% make the meshgrid for derivative
L=kron(Iy,Dx)+kron(Dy,Ix);

% make D with xy domain
% By playing around find that the rash is on 
% A(135:165,155:185)
D = zeros(nx,ny);
for j = 135:165
    for jj = 155:185
        D(j,jj) = 1;
    end
end
tspan = [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08];
Dc = 0.001;
D = Dc*D;
Ds = reshape(D, nx*ny, 1);
R = A(:,:,1);
G = A(:,:,2);
B = A(:,:,3);
Rn0 = reshape(R,nx*ny,1);
Gn0 = reshape(G,nx*ny,1);
Bn0 = reshape(B,nx*ny,1);
[t,R_sol] = ode45('image_rhs',tspan, Rn0,[],L,Ds);
[t,G_sol] = ode45('image_rhs',tspan, Gn0,[],L,Ds);
[t,B_sol] = ode45('image_rhs',tspan, Bn0,[],L,Ds);

figure(1)
A_sol = zeros(size(A));
for j = 1:9
    A_sol = zeros(size(A));
    subplot(3,3,j)
    Rc = reshape(R_sol(j,:),nx,ny);
    Gc = reshape(G_sol(j,:),nx,ny);
    Bc = reshape(B_sol(j,:),nx,ny);
    A_sol(:,:,1) = Rc;
    A_sol(:,:,2) = Gc;
    A_sol(:,:,3) = Bc;
    
    imshow(uint8(A_sol));
    title(['Time: ', num2str(tspan(j))]);
end