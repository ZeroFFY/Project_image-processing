
function [imgn,rot] = rotateimage(image_close,img_num)
for jj=1:1:img_num
    [M,N] = size(image_close{jj});
    IMG = image_close{jj}
    roi_flag = 0;
    while (roi_flag == 0) 
        disp('Choose ROI corner...');
        figure(10); 
        imshow(IMG); 
        title(['IMG', num2str(jj)]); 
        hold on;
        roi_x = [];
        roi_y = [];
        for count = 1:4
            [roi_xi, roi_yi] = ginput(1);
            plot(roi_xi, roi_yi,'g+','linewidth',2);
            roi_x = [roi_x;roi_xi];
            roi_y = [roi_y;roi_yi];
            plot(roi_x,roi_y,'g-','linewidth',2);
            drawnow;
            dot(count,1)=roi_xi;
            dot(count,2)=roi_yi;
        end
        plot([roi_x;roi_x(1)],[roi_y;roi_y(1)],'g-','linewidth',2);
        drawnow;
        hold off;
        roi_flag_ = input('Sure ROI?: ([]=yes, other=no) ', 's');
        if isempty(roi_flag_)
            roi_flag = 1;
        end
    end
    w=round(sqrt((dot(1,1)-dot(2,1))^2+(dot(1,2)-dot(2,2))^2));     %��ԭ�ı��λ���¾��ο�
    h=round(sqrt((dot(1,1)-dot(4,1))^2+(dot(1,2)-dot(4,2))^2));     %��ԭ�ı��λ���¾��θ�
    y=[dot(1,1) dot(2,1) dot(4,1) dot(3,1)];        %�ĸ�ԭ����
    x=[dot(1,2) dot(2,2) dot(4,2) dot(3,2)];%�����ԭͼ���Ǿ��Σ���ͼ���Ǵ�dot��ȡ�õĵ���ɵ������ı���.:)
    Y=[dot(1,1) dot(1,1) dot(1,1)+h dot(1,1)+h];     
    X=[dot(1,2) dot(1,2)+w dot(1,2) dot(1,2)+w];
    B=[X(1) Y(1) X(2) Y(2) X(4) Y(4) X(3) Y(3)]';   %�任����ĸ����㣬�����ұߵ�ֵ
%�����ⷽ���飬���̵�ϵ��
   A=[x(1) y(1) 1 0 0 0 -X(1)*x(1) -X(1)*y(1);             
      0 0 0 x(1) y(1) 1 -Y(1)*x(1) -Y(1)*y(1);
      x(2) y(2) 1 0 0 0 -X(2)*x(2) -X(2)*y(2);
      0 0 0 x(2) y(2) 1 -Y(2)*x(2) -Y(2)*y(2);
      x(4) y(4) 1 0 0 0 -X(4)*x(4) -X(4)*y(4);
      0 0 0 x(4) y(4) 1 -Y(4)*x(4) -Y(4)*y(4);
      x(3) y(3) 1 0 0 0 -X(3)*x(3) -X(3)*y(3);
      0 0 0 x(3) y(3) 1 -Y(3)*x(3) -Y(3)*y(3)];
  fa=inv(A)*B;        %���ĵ���õķ��̵Ľ⣬Ҳ��ȫ�ֱ任ϵ��
  a=fa(1);b=fa(2);c=fa(3);
  d=fa(4);e=fa(5);f=fa(6);
  g=fa(7);h=fa(8);
  rot{jj}=[d e f;
       a b c;
       g h 1];        %��ʽ�е�һ������x,Matlab��һ����ʾy�������Ҿ���1,2�л�����
  pix1=rot{jj}*[1 1 1]'/(g*1+h*1+1);  %�任��ͼ�����ϵ�
  pix2=rot{jj}*[1 N 1]'/(g*1+h*N+1);  %�任��ͼ�����ϵ�
  pix3=rot{jj}*[M 1 1]'/(g*M+h*1+1);  %�任��ͼ�����µ�
  pix4=rot{jj}*[M N 1]'/(g*M+h*N+1);  %�任��ͼ�����µ�

  height=round(max([pix1(1) pix2(1) pix3(1) pix4(1)])-min([pix1(1) pix2(1) pix3(1) pix4(1)]));     %�任��ͼ��ĸ߶�
  width=round(max([pix1(2) pix2(2) pix3(2) pix4(2)])-min([pix1(2) pix2(2) pix3(2) pix4(2)]));      %�任��ͼ��Ŀ��
  imgn{jj}=zeros(height,width);

  delta_y=round(abs(min([pix1(1) pix2(1) pix3(1) pix4(1)])));            %ȡ��y����ĸ��ᳬ����ƫ����
  delta_x=round(abs(min([pix1(2) pix2(2) pix3(2) pix4(2)])));            %ȡ��x����ĸ��ᳬ����ƫ����
  inv_rot{jj}=inv(rot{jj});

  for i = 1-delta_y:height-delta_y                        %�ӱ任ͼ���з���Ѱ��ԭͼ��ĵ㣬������ֿն�������ת�Ŵ�ԭ��һ��
      for j = 1-delta_x:width-delta_x
          pix=inv_rot{jj}*[i j 1]';       %��ԭͼ�������꣬��Ϊ[YW XW W]=fa*[y x 1],�������������[YW XW W],W=gy+hx+1;
          pix=inv([g*pix(1)-1 h*pix(1);g*pix(2) h*pix(2)-1])*[-pix(1) -pix(2)]'; %�൱�ڽ�[pix(1)*(gy+hx+1) pix(2)*(gy+hx+1)]=[y x],����һ�����̣���y��x�����pix=[y x];
          if pix(1)>=0.5 && pix(2)>=0.5 && pix(1)<=M && pix(2)<=N
            imgn{jj}(i+delta_y,j+delta_x)=IMG(round(pix(1)),round(pix(2)));     %���ڽ���ֵ,Ҳ������˫���Ի�˫������ֵ
          end
      end
  end  
end
  figure();
  paint(imgn,img_num);
  
end
