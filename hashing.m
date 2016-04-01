function[J]=hashing(x)

[r c]=size(x);
bs=8; % Block Size (8x8)
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=x((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(c/bs);
end
f=1;
z=Block(:,:,f);
d=Block(:,:,f+1);
J=bitxor(z,d); 
for f=3:128 
     z=Block(:,:,f);      
     J=bitxor(J,z);
end
 return   
    