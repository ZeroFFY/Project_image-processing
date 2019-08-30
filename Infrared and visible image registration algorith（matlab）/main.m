close all;
clear all;
% 载入图像  
%%%发现只有匹配点对遍布在图像中，匹配效果才会变好  
I11=(imread('image1/qa1.bmp'));%红外图像  
I22=(imread('image1/qa2.bmp'));%可见光图像  

[m1,n1,o1] = size(I11);
[m2,n2,o2] = size(I22);
if m1<m2;
    I22_t = imresize(I22,[m1 n1]);
    I22 = I22_t;
else
    I11_t = imresize(I11,[m1 n2]);
    I11 = I11_t;
end

I10=rgb2gray(I11);  
I20=rgb2gray(I22);  
I1=imadjust(I10);  
I2=imadjust(I20);  
image_close{1} = I1;
image_close{2} =I2;
img_num = 2;
[imgn,rot] = rotateimage(image_close,img_num);
figure;
imshow(imgn{1},[]);
I1 = imgn{1};
I2 = imgn{2};
%边缘检测
I1 = im2double(edge(I1,'canny',0.2));   
I2 = im2double(edge(I2,'canny',0.2));  
se1=strel('square',1);

I1=imopen(I1,se1);%开运算去除散点
I1=imclose(I1,se1);%闭运算连接散点边缘
I2=imopen(I2,se1);%开运算去除散点
I2=imclose(I2,se1);%闭运算连接散点边缘
figure,imshow(I1);  
figure,imshow(I2);  
  
% 获取特征点
Options.upright=true;  
Options.tresh=0.0001;  
Ipts1=OpenSurf(I1,Options);  %1*268  
Ipts2=OpenSurf(I2,Options);  

% 将特征点描述子放入一个矩阵 
D1 = reshape([Ipts1.descriptor],64,[]);  %64*268  
D2 = reshape([Ipts2.descriptor],64,[]);  
  
% 找到最佳匹配
err=zeros(1,length(Ipts1));     
cor1=1:length(Ipts1);  
cor2=zeros(1,length(Ipts1));  
for i=1:length(Ipts1)  
    distance=sum((D2-repmat(D1(:,i),[1 length(Ipts2)])).^2,1);%用D2的每一列分别去减D1的每一列  
    [err(i),cor2(i)]=min(distance);  
end  
  
%向量距离排序  
[err, ind]=sort(err);  
cor1=cor1(ind);  
cor2=cor2(ind);  
  
% 找到向量坐标的最佳匹配
Pos1=[[Ipts1(cor1).y]',[Ipts1(cor1).x]'];  
Pos2=[[Ipts2(cor2).y]',[Ipts2(cor2).x]'];  
Pos1_1=Pos1;  
Pos2_1=Pos2;  
Pos1(:,3)=1; Pos2(:,3)=1;Pos1_1(:,3)=1; Pos2_1(:,3)=1;  
Pos1=Pos1(1:20,:);  
Pos2=Pos2(1:20,:);  
Pos11=zeros(size(Pos1));  
Pos22=zeros(size(Pos2));  

% 计算亲和矩阵
for i=1:20  
     k(i)=Pos2(i,2)/(Pos2(i,1)-Pos1(i,1));  
end  
tic;  
%%  
%自适应确定k的范围  
positive_num=find(k>0);  
negative_num=find(k<0);  
if length(positive_num)>length(negative_num)  
    flag=1;  
    num_err=negative_num;  
    num_correct=positive_num;  
else  
    flag=0;  
    num_err=positive_num;  
    num_correct=negative_num;  
end  
k(num_err)=[];  
k_1=round(k);  
k_2=unique(k_1);%求出k_1有哪些数，剔除重复数据  
temp=zeros(length(k_2),2);%第一列存数，第二列存该数的个数%后边还需要对个数排序  
for i=1:length(k_2)  
    temp_num=0;  
    temp(i,1)=k_2(i);  
   for j=1:length(k)  
       if temp(i,1)==k_1(j)  
          temp_num=temp_num+1;   
       end  
   end  
   temp(i,2)=temp_num;  
end  
%temp  
  
[temp1,temp1_location]=sort(temp(:,2),'descend');
xu=temp1(2);  
temp=temp(temp1_location); 
k_1=zeros(size(1,2));  
k_1(1)=temp(1);  
k_1(2)=temp(2);  
k_1=sort(k_1,'descend');  
k_11=k_1(1,1)+0.5;  
k_12=k_1(1,1)-0.5;  
k_21=k_1(1,2)-0.5; 
k_22=k_1(1,2)+0.5; 
count=0;  
  
for i=1:20  
     k=Pos2(i,2)/(Pos2(i,1)-Pos1(i,1));   %%%%可不可以把数据归一化  
     if xu>1  
if    (k<k_11)&&(k>k_12) || (k<k_22)&&(k>k_21)      
        count=count+1;  
        Pos11(count,:)=Pos1(i,:);   
        Pos22(count,:)=Pos2(i,:);   
end    
     else  
         if   (k<k_11)&&(k>k_12)                                            
       count=count+1;  
        Pos11(count,:)=Pos1(i,:);   
        Pos22(count,:)=Pos2(i,:);   
         end  
     end  
end  
%%  
%对重复出现的点进行剔除  
[SIZEm,SIZEn]=size(Pos22);  
num=zeros(1,SIZEm);   
for i=1:SIZEm  
    for j=1:SIZEm  
    if Pos22(i,1)==Pos22(j,1)  
        num(i)=num(i)+1;  
    end  
    end  
end  
error=find(num>1);%错误点的位置  
Pos11(error,:)=[];   
Pos22(error,:)=[];  
error1=find(Pos22(:,1)==0);  
Pos11(error1,:)=[];  
Pos22(error1,:)=[];  
M=Pos11'/Pos22';  
  
  
%%%%初步确定变换矩阵后对所有的初步数据进行遍历，最后再去除错误点（重复出现的、有多余0的）  
Pos11_1=zeros(size(Pos1_1));  
Pos22_1=zeros(size(Pos2_1));  
%pos_count=1;  
for i=21:size(Pos1_1,1)  
    delta=abs(M*Pos2_1(i,:)'-Pos1_1(i,:)');  
    if delta<[30;30;1]  
        Pos11_1(i,:)= Pos1_1(i,:);  
        Pos22_1(i,:)= Pos2_1(i,:);  
    end  
end  
%%  
[tem_m,tem_n]=size(Pos11);%把最初20个数里得到的几个点拷贝到现在的数组Pos11_1、Pos22_1里  
Pos11_1(1:tem_m,:)=Pos11(1:tem_m,:);  
Pos22_1(1:tem_m,:)=Pos22(1:tem_m,:);  
[tem_SIZEm,tem_SIZEn]=size(Pos22_1);  
num=zeros(1,tem_SIZEm);   
for i=1:tem_SIZEm  
    for j=1:tem_SIZEm  
    if Pos22_1(i,1)==Pos22_1(j,1)  
        num(i)=num(i)+1;  
    end  
    end  
end  
error=find(num>1);%错误点的位置  
Pos11_1(error,:)=[];   
Pos22_1(error,:)=[];  
error1=find(Pos22_1(:,1)==0);  
Pos11_1(error1,:)=[];  
Pos22_1(error1,:)=[];  
%M=Pos11_1'/Pos22_1'  
  
%从前十个比较准确的匹配点对中随机选取三个点对形成M,再类似于RANSAC运算，可靠性优于前者。  
TEN1=Pos1_1(1:10,:,:);%红外图像中前10个点  
TEN2=Pos2_1(1:10,:,:);%可见光图像中前10个点  
  
for i=1:10  
     k(i)=(TEN2(i,2)+240-TEN1(i,2))/(TEN2(i,1)-TEN1(i,1));  
end  
if flag==1  
    TEN_location=find(k<0);  
    TEN1(TEN_location,:,:)=[];   
    TEN2(TEN_location,:,:)=[];   
else   
    TEN_location=find(k>0);  
    TEN1(TEN_location,:,:)=[];   
    TEN2(TEN_location,:,:)=[];   
end  

%上面是把斜率为负的一些点先排除掉   得到TEN1和TEN2两组纯点对  
TEN1_num=size(TEN1,1);  
TEN_COUNT1=0;  
TEN_COUNT2=0;  
pp=0;  
for i=1:TEN1_num  
    for j=1:TEN1_num  
        for k=1:TEN1_num  
            if (i>=j)||(i>=k)||(j>=k)  
                continue;  
            end  
            pp=pp+1;  
            temp_TEN1=TEN1([i j k],:,:);%随机取的三个点  
            temp_TEN2=TEN2([i j k],:,:);  
            temp_M = temp_TEN1'* pinv(temp_TEN2'); 
 
            %对上面求得数据进行遍历  
            TEN_COUNT2=0;  
            for v=1:size(Pos11_1,1)  
            alpha=abs(temp_M*Pos22_1(v,:,:)'-Pos11_1(v,:,:)');  
            if (alpha<[2;2;1])  
                TEN_COUNT2=TEN_COUNT2+1;  
            end  
            end             
  
            if (TEN_COUNT2>=TEN_COUNT1)  
                TEN_COUNT1=TEN_COUNT2;  
                best_i=i;  
                best_j=j;  
                best_k=k;  
            end     
        end  
    end  
end  
w1_1=zeros(size(Pos11_1));  
w1_2=zeros(size(Pos11_1));  
temp_TEN1=TEN1([best_i best_j best_k],:,:);  
temp_TEN2=TEN2([best_i best_j best_k],:,:);  
M=temp_TEN1'/temp_TEN2';  
w1_1(1:3,:,:)=TEN1([best_i best_j best_k],:,:);  
w1_2(1:3,:,:)=TEN2([best_i best_j best_k],:,:);  
count=4;  
   for i=1:size(Pos11_1,1)  
            alpha=abs(M*Pos22_1(i,:,:)'-Pos11_1(i,:,:)');  
            if (alpha<[2;2;1])             
                w1_1(count,:,:)=Pos11_1(i,:,:);  
                w1_2(count,:,:)=Pos22_1(i,:,:);  
                count=count+1;  
            end  
   end   
toc;  
M=w1_1'/w1_2';  
u=find(w1_1(:,1,1)==0);  
w1_1=w1_1(1:u(1)-1,:);  
w1_2=w1_2(1:u(1)-1,:);  
w1_3=(M*w1_2')';  
RMSE_1=0;  
for i=1:size(w1_1,1)  
    RMSE_1=RMSE_1+((w1_1(i,1)-w1_3(i,1))^2)+((w1_1(i,2)-w1_3(i,2))^2);  
end  
 RMSE=sqrt(RMSE_1/(u(1)-1))  
 %RMSE=sqrt(RMSE_1/size(w1_1,1))  

% 图像需要一样大  
I = zeros([size(I11,1) size(I11,2)*2 size(I11,3)]);  
I(:,1:size(I11,2),:)=I11; I(:,size(I11,2)+1:size(I11,2)+size(I22,2),:)=I22;  
figure, imshow(rgb2gray(uint8(I))); hold on;  
  
  
% 显示最佳匹配 
plot([w1_1(:,2) w1_2(:,2)+size(I1,2)]',[w1_1(:,1) w1_2(:,1)]','-');  
plot([w1_1(:,2) w1_2(:,2)+size(I1,2)]',[w1_1(:,1) w1_2(:,1)]','o');  
  
  
% 融合图像 
I1_warped=affine_warp(I11,M,'bicubic');    
  
% 展示结果
figure,  
subplot(2,2,1), imshow(rgb2gray(I11));title('红外图像');  
subplot(2,2,2), imshow(rgb2gray(I22));title('可见光图像');  
subplot(2,2,3), imshow(rgb2gray(uint8(I1_warped)));title('配准后图像');  
subplot(2,2,4),imshow(rgb2gray(0.5*I22+0.5*uint8(I1_warped)));title('融合图像');  
 
imwrite(I11,'image/1.png');  
imwrite(I22,'image/2.png');  
imwrite(uint8(I1_warped),'image/3.png');  
imwrite(0.5*I22+0.5*uint8(I1_warped),'image/4.png'); 