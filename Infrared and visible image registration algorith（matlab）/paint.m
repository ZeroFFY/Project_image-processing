function paint(image,img_num,x)
n=ceil(sqrt(img_num));
figure;
if img_num<=n*(n-1)
    m=n-1;
else
    m=n;
end
if img_num>2
    for j=1:1:img_num
        subplot(m,n,j),imshow(image{j});
    end
elseif img_num==2
    subplot(1,2,1),imshow(image{1})
    subplot(1,2,2),imshow(image{2})
else 
    imshow(image{1})
end
title('x');
end
