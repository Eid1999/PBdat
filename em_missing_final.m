%Escolher imagem
%im=imresize(rgb2gray(imread('image_screen.jpg')),.5);
im=imread('lena.gif');
im=double(im)/255;
W=im;
M=rand(size(W))<.55;
row=ones(1,size(W,2));
col=ones(size(W,1),1);
%row(1:5)=0;
%col(1:5)=0;
%M=toeplitz(row,col);
M(100:120,100:120)=0;%bloco missing
W=W.*M;

%%
%---------inicializa
clf
indsv=find(M(:)==1);
indsmiss=find(M(:)==0);
[row col]=size(W);
What=full(W);
M=full(M);
for i=1:size(What,1),
    What(i,M(i,:)<1)=mean(W(i,M(i,:)>0));
end
%-----visualiza
figure(1);
imagesc([im What W]);colormap(gray);
figure(2);plot(svd(im));

%----------iteracoes
%%
Dk=zeros(size(W));
Dk=M.*W+(1-M).*What;
%Dk=M.*W;%inicializaçao radical
erro=Inf;
iter=0;
maxiter=1000;
maxerror=1;
r=50;
while erro >maxerror & iter<maxiter,
    iter=iter+1;
    %Dkm=Dk;%Dkm(indsv)=0;
    [u s v]=svd(Dk,'econ');

    What=u(:,1:r)*s(1:r,1:r)*v(:,1:r)';
    What(What(:)<0)=0;%Para o caso  de imagens, project primeiro ortante
    What(What(:)>1)=1;
    Dk=M.*W+(1-M).*What;
    errov=(M.*(W-What));
    %subplot(121);
    figure(1);imagesc([im Dk W]);colormap(gray);
    %subplot(122);
    figure(2);mesh(abs(errov));axis ij
    erro=norm(errov,'fro');
    fprintf('erro %g  iter %g \n',erro,iter);
    drawnow;
    if iter <5,pause;end % para visualizar primeiras iterações

end
%%
%Pseudo inversas
Dk=zeros(size(W));
Dk=M.*W+(1-M).*What;

erro=Inf;
iter=0;
maxiter=3000;
maxerror=.01;
r=10;
[u s v]=svd(Dk,'econ');
A=u(:,1:r)*sqrt(s(1:r,1:r));
Bt=sqrt(s(1:r,1:r))*v(:,1:r)';
while erro >maxerror & iter<maxiter,
    iter=iter+1;
    %Dkm=Dk;%Dkm(indsv)=0;
    A=Dk*Bt'*inv(Bt*Bt');
    Bt=inv(A'*A)*A'*Dk;
    What=A*Bt;
    What(What(:)<0)=0;%Para o caso  de imagens, project primeiro ortante
    What(What(:)>1)=1;
    Dk=M.*Dk+(1-M).*What;
    errov=(M.*(W-What));
    %subplot(121);
    figure(1);imagesc([im Dk W]);colormap(gray);
    %subplot(122);
    figure(2);mesh(abs(errov));axis ij
    erro=norm(errov,'fro');
    fprintf('erro %g  iter %g \n',erro,iter);
    drawnow;
    if iter <5,pause;end % para visualizar primeiras iterações

end