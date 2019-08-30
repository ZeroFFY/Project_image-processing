clear;
clc;
close all;
% input and read picture 
x=input('please input the first initial name of picutre:','s');
y=input('please input the image format(jpg:j,bmp:b,png:p):','s');
while(1)
if y=='b'||y=='j'||y=='p'
    break;
else
    y=input('please input the image format(jpg:j,bmp:b,png:p):','s');
end
end
if y=='b'
    img_path_list = dir(strcat(x,'*.bmp'));
elseif y=='j'
    img_path_list = dir(strcat(x,'*.jpg'));
elseif y=='p'
    img_path_list = dir(strcat(x,'*.png'));
end
img_num = length(img_path_list);
%��ͼ���Ϊ�Ҷ�ͼ
for j=1:1:img_num
     I= imread(img_path_list(j).name);
     gdata=rgb2gray(I);
    image{j}=gdata;
end
paint(image,img_num);
%k-means ����ͼ��
 k=input('please input the k-means K(positive integer):');
 for i =1:img_num
 ima = image{i};
[mu,mask]=kmeans(ima,k);
mask_img{i} = mask;
 end
paint(mask_img,img_num);title(['k=',num2str(k)]);
se1=strel('square',2);
se2=strel('square',7);
z1=input('please input the canny parameter��0-1��:');%��ȡ��Ե
for jj =1:img_num
    mask_img{jj}(mask_img{jj}<(k-1))=1;
    fo{jj}=imclose(mask_img{jj},se1); 
    imageedge{jj} = edge(fo{jj},'canny',z1);%canny������ȡ��Ե
    fo{jj}=imclose(imageedge{jj},se1);%����������ɢ���Ե
    fo{jj} =imfill(fo{jj},'holes');%������ӵı�Ե
    fo{jj}=imopen(fo{jj},se2);%������ȥ��ɢ��
end
paint(fo,img_num);title('�����ͼ��');
