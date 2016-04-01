%// Load an image
Orig =imread('C:\Users\Gaurav\Desktop\bpcs_project\c.bmp');
W=imresize(imread('C:\Users\Gaurav\Desktop\bpcs_project\lena.jpg'),[64 64]);
R=imresize(imread('C:\Users\Gaurav\Desktop\bpcs_project\Wc.png'),[8 8]);
%// Transform
Orig_T = dct2(Orig);
%// Split between high- and low-frequency in the spectrum (*)
cutoff = round(0.5 * 256);
High_T = fliplr(tril(fliplr(Orig_T), cutoff)); %fliplr=flip matrix from left to right
% tril=lower triangular matrix
Low_T = Orig_T - High_T;
%// Transform back
High = idct2(High_T);
Low = idct2(Low_T);

%// Plot results
figure, colormap gray
subplot(3,2,1), imagesc(Orig), title('Original'), axis square, colorbar % imagesc=Display image using scaled colors
subplot(3,2,2), imagesc(log(abs(Orig_T))), title('log(DCT(Original))'),  axis square, colorbar
%colorbar displays a vertical colorbar to the right of the current axes. Colorbars display the current colormap and indicate the mapping 
%of data values into the colormap

subplot(3,2,3), imagesc(log(abs(Low_T))), title('log(DCT(LF))'), axis square, colorbar
subplot(3,2,4), imagesc(log(abs(High_T))), title('log(DCT(HF))'), axis square, colorbar

subplot(3,2,5), imagesc(Low), title('LF'), axis square, colorbar
subplot(3,2,6), imagesc(High), title('HF'), axis square, colorbar

k=256;
A1=zeros(129,256);
f=1;
s=1;

for i=1:256
    for j=1:k
        A1(f,s)=Low(i,j);
        s=s+1;
        if s==257
            s=1;
            f=f+1;
        end
        if f==130
            break;
        end
    end
    k=k-1;
end    
A1=imresize(A1,[64 64]);
A1=round(A1);%Dispaying matrix A.
size(A1)

B1=[W,A1];
size(B1)

L=hashing(B1);
L

p=17;
q=11;
n=p*q;
Fi=(p-1)*(q-1);
e=7;

for i=1:Fi
    s1=i*e;
    s2=mod(s1,Fi);
    if s2==1
        d=i;
        break;
    end
end
d

%public key={e,n}={7,187}
%private key={d,n}={d,187}
%performing encryption
E1=zeros(8,8);
for i=1:8
    for j=1:8
        o=L(i,j);
        pr=power(o,e);
        E1(i,j)=mod(pr,n);
    end
end    
E1=im2bw(E1);

%BPCS Technique

x=Orig;

A=zeros(256,256);
for i=1:256
A(i,1)=x(i,1);
end 
for i=1:256
for j=2:256 
    A(i,j)=bitxor(x(i,j-1),x(i,j)); 
end 
end 
figure,subplot(1,2,1);
imshow(x);
title('Original Image');
subplot(1,2,2);
imshow(A);
title('Image in CGC System');

%Segment each bit-plane of the dummy image into informative and noise-like regions by using a threshold value. A typical value is ?0 = 0.3. 
%Bit plane slicing
B=bitget(A,1);
C=bitget(A,2);
D=bitget(A,3);
E=bitget(A,4);
F=bitget(A,5);
G=bitget(A,6);
H=bitget(A,7);
I=bitget(A,8);

figure,
subplot(2,4,1);imshow(logical(B));title('Bitplane 1');     %Show Bitplane 1 
subplot(2,4,2);imshow(logical(C));title('Bitplane 2');     %Show Bitplane 2
subplot(2,4,3);imshow(logical(D));title('Bitplane 3');     %Show Bitplane 3
subplot(2,4,4);imshow(logical(E));title('Bitplane 4');     %Show Bitplane 4
subplot(2,4,5);imshow(logical(F));title('Bitplane 5');     %Show Bitplane 5
subplot(2,4,6);imshow(logical(G));title('Bitplane 6');     %Show Bitplane 6
subplot(2,4,7);imshow(logical(H));title('Bitplane 7');     %Show Bitplane 7
subplot(2,4,8);imshow(logical(I));title('Bitplane 8');     %Show Bitplane 8

%Segmenting the bit planes into informative and noise like part.
[r c]=size(B);
bs=8; % Block Size (8x8)
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=B((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z=zeros(1024,1);
z1=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z(k)=c9;    z1(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z(i)=z1(i)/z(i); 
end

%Embedding watermark(W)+signature(E1) in informative part of image
l=1;
for i=1:8:64
    for j=1:8:64
       J{l}=W(i:i+7,j:j+7);
       l=l+1;
    end
end

bno=zeros(1,65);    %contain block numbers where signature+watermark blocks are embedded
lp=1;
alpha=0.3;
for i=1:1024
    if z(i)<alpha
        if lp==65
            break;
        end    
        bno(lp)=i;
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        B(m1:m2,k1:k2)=J{lp};
        lp=lp+1;
    end
end

g=bno(64);

s=1;
for i=(g+1):1024
    if z(i)<alpha
        if s==2
            break;
        end   
        bno(lp)=i;
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        B(m1:m2,k1:k2)=E1;
        s=s+1;
        lp=lp+1;
    end
end

alpha=0.3;  c=0;
display('Bit Plane 1');
for i=1:1024            %Bit Plane 1
    if z(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

%Group the secret file ie. a binary image into a series of secret blocks.
% There would be 64 secret blocks of 8X8 size for a secret file of 64X64.
img=imresize(imread('binary.jpg'),[64 64]);
img=im2bw(img);
figure,subplot(13,5,1);
imshow(img);
l=1;
k=2;
i=1;
j=1;
title('8X8 blocks of secret file');
for i=1:8:64
    for j=1:8:64
       J{l}=img(i:i+7,j:j+7);
       subplot(13,5,k);
       k=k+1;
       imshow(J{l});
       l=l+1;
    end
end



%Embedding secret file blocks in the noise blocks of bit planes.
s=1;
bno1=zeros(1,64); %contain block numbers where secret blocks are embedded
l=1;
for i=1:1024
    if z(i)>alpha
        if s==65
            break;
        end
        bno1(l)=i;
        l=l+1;
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        B(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

conmap=zeros(1024,1);
flag=0;
display('Computing conjugate of all informative blocks of Bit Plane 1');
for i=1:1024    %Bit Plane 1
   if z(i)<alpha
       for t1=1:65
           if i==bno[t1]
           flag=1;
           break;
           end
       end
       if flag==0
       conmap(i)=1;
       m1=floor((i-1)/32)*8+1;
       m2=m1+7;
       k3=mod((i-1),32);
       k1=floor(k3)*8+1; 
       k2=k1+7;
        u=1;
      for m=m1:m2
       v=1;  
      for k=k1:k2
      B(k,m)=bitxor(B(k,m),R(u,v)); 
      v=v+1;
      end 
      u=u+1;
      end 
      end
    end 
end   

% ***********************ENTROPY calculation*********************

% The entropy is calculated of each 8x8 block in every bit plane.
% Then we will find out the maximum entropy from each bit plane.
% And get the maximum entropy from the maximums.


J{1}=B;
J{2}=C;
J{3}=D;
J{4}=E;
J{5}=F;
J{6}=G;
J{7}=H;
J{8}=I;

row=size(x,1)/8;
col=size(x,2)/8;

maxresult=zeros(1,8)
c=1;
for i=1:8
    curr_plane=J{i};
    blocks=mat2cell(curr_plane,ones(1,row)*8,ones(1,col)*8); %Dividing current plane into 8x8 block 
    for bi=1:size(blocks,1)
      for bj=1:size(blocks,2)
         my_8x8_block=blocks{bi,bj};
         a(c)=entropy(nanmean(my_8x8_block)); %Calculating entropy of block
         Q{c}=a(c);
         c=c+1;
      end
    end 
    
    % Extracting the maximum entropy block for each bit plane.
     c=1;
     maxim=0;
    for bi=1:size(blocks,1)
       for bj=1:size(blocks,2)
           if(a(c)>maxim)
               maxim=a(c);
               blocknumber=c;
           end
           c=c+1;  
       end
    end
    maxresult(i)=maxim;
    block(i)=blocknumber;
    P{i}=Q;
    c=1;
end

%Extracting the maximum entropy and its corresponding bit plane from the maximums. 

maxi=0;
for i=1:8
   if(maxresult(i)>maxi)
       maxi=maxresult(i)
       corresponding_block=block(i)
       curr_plane=i;
   end
end



display(maxresult)
display(block)

display(maxi)
display(corresponding_block)
display(curr_plane)

%******************************END******************************

% %************EMBEDDING THE CONJUGATION MAP********************

% Embedding the conjugation map IN corresponding_block i.e.
% the block which have maximum entropy in curr_plane i.e. bit plane number.
n=1;
for i=1:16
    j=n;
    
    m1=floor((corresponding_block-1)/32)*8;
    m2=m1+7;
    k3=mod((corresponding_block-1),32);
    k1=floor(k3)*8+1;
    k2=k1+7;
    for m=m1:m2
        for p=k1:k2
           curr_plane(m,p)=conmap(j);
           j=j+1;
        end
    end
    corresponding_block=corresponding_block+1;
    n=j-1;
end
%*******************END***********************

%Image reconstruction
M=zeros(size(A));
M=bitset(M,8,bitget(A,8));
M=bitset(M,7,bitget(A,7));
M=bitset(M,6,bitget(A,6));
M=bitset(M,5,bitget(A,5));
M=bitset(M,4,bitget(A,4));
M=bitset(M,3,bitget(A,3));
M=bitset(M,2,bitget(A,2));
M=bitset(M,1,bitget(A,1));
M=uint8(M);
figure,imshow(M);

%Converting CGC to PBC
xy=zeros(256,256);
for i=1:256
xy(i,1)=M(i,1); 
end 
for i=1:256
    for j=2:256
        xy(i,j)=bitxor(M(i,j),xy(i,j-1)); 
    end 
end 
pbc=uint8(xy);
figure,subplot(1,2,1);imshow(Orig);
title('Original Image');
subplot(1,2,2);imshow(pbc);
title(' Image after embedding secret file ');