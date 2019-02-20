%Function used to compute Haralick Features out of an 256X256 image.
%Reference:Robert M. Haralick, K. Shanmugam, Its'Hak Dinstein, “Textural Features for Image Classification”,
%IEEE Journals & Magazines, vol. SMC-3, 1973, pp. 610-621.
%@version 1.0
%@author Veinstin Furtado <vrfurtado@mun.ca>

function [f1, f2, f3, f5, f6, f7, f8, f9] = generateImageFeatures(input_image)
gclm_0 = graycomatrix(input_image,'NumLevels',256,'offset', [0 1],'GrayLimits',[],'Symmetric', true);
gclm_45 = graycomatrix(input_image,'NumLevels',256, 'offset', [-1 1],'GrayLimits',[],'Symmetric', true);
gclm_90 = graycomatrix(input_image,'NumLevels',256, 'offset', [-1 0],'GrayLimits',[],'Symmetric', true);
gclm_135 = graycomatrix(input_image,'NumLevels',256, 'offset', [-1 -1],'GrayLimits',[],'Symmetric', true);
gclm_0_norm = gclm_0/sum(gclm_0(:));
gclm_45_norm = gclm_45/sum(gclm_45(:));
gclm_90_norm = gclm_90/sum(gclm_90(:));
gclm_135_norm = gclm_135/sum(gclm_135(:));

%Required for calculating the features
Ng = 256;
px_0 = sum(gclm_0_norm,2);
px_45 = sum(gclm_45_norm,2);
px_90 = sum(gclm_90_norm,2);
px_135 = sum(gclm_135_norm,2);
mux_0 = mean(px_0,'all');
mux_45 = mean(px_45,'all');
mux_90 = mean(px_90,'all');
mux_135 = mean(px_135,'all');
stdx_0 = std(px_0);
stdx_45 = std(px_45);
stdx_90 = std(px_90);
stdx_135 = std(px_135);
py_0 = sum(gclm_0_norm,1);
py_45 = sum(gclm_45_norm,1);
py_90 = sum(gclm_90_norm,1);
py_135 = sum(gclm_135_norm,1);
muy_0 = mean(py_0,'all');
muy_45 = mean(py_45,'all');
muy_90 = mean(py_90,'all');
muy_135 = mean(py_135,'all');
stdy_0 = std(py_0);
stdy_45 = std(py_45);
stdy_90 = std(py_90);
stdy_135 = std(py_135);
pxpy_0 = zeros(1,2*Ng);
pxpy_45 = zeros(1,2*Ng);
pxpy_90 = zeros(1,2*Ng);
pxpy_135 = zeros(1,2*Ng);
for i=1:Ng
   for j=1:Ng
        pxpy_0(i+j) = pxpy_0(i+j) + gclm_0_norm(i,j);
        pxpy_45(i+j) = pxpy_45(i+j) + gclm_45_norm(i,j);
        pxpy_90(i+j) = pxpy_90(i+j) + gclm_90_norm(i,j);
        pxpy_135(i+j) = pxpy_135(i+j) + gclm_135_norm(i,j);
    end
end
pxmy_0 = zeros(1,Ng);
pxmy_45 = zeros(1,Ng);
pxmy_90 = zeros(1,Ng);
pxmy_135 = zeros(1,Ng);
for i=1:Ng
    for j=1:Ng
        pxmy_0(abs(i-j)+1) = pxmy_0(abs(i-j)+1) + gclm_0_norm(i,j);
        pxmy_45(abs(i-j)+1) = pxmy_45(abs(i-j)+1) + gclm_45_norm(i,j);
        pxmy_90(abs(i-j)+1) = pxmy_90(abs(i-j)+1) + gclm_90_norm(i,j);
        pxmy_135(abs(i-j)+1) = pxmy_135(abs(i-j)+1) + gclm_135_norm(i,j);
    end
end

%f1:Angular Second Moment
f1_0 = 0;
f1_45 = 0;
f1_90 = 0;
f1_135 = 0;
for i=1:256
    for j=1:256
        f1_0 = f1_0 + ((gclm_0_norm(i,j))*(gclm_0_norm(i,j)));
        f1_45 = f1_45 + ((gclm_45_norm(i,j))*(gclm_45_norm(i,j)));
        f1_90 = f1_90 + ((gclm_90_norm(i,j))*(gclm_90_norm(i,j)));
        f1_135 = f1_135 + ((gclm_135_norm(i,j))*(gclm_135_norm(i,j)));
    end
end
f1 = (f1_0 + f1_45 + f1_90 + f1_135)/4;

%f2:Contrast
f2_0 = 0;
f2_45 = 0;
f2_90 = 0;
f2_135 = 0;
for k=1:Ng
    f2_0 = f2_0 + ((k*k)*pxmy_0(k));
    f2_45 = f2_45 + ((k*k)*pxmy_45(k));
    f2_90 = f2_90 + ((k*k)*pxmy_90(k));
    f2_135 = f2_135 + ((k*k)*pxmy_135(k));
end
f2 = (f2_0 + f2_45 + f2_90 + f2_135)/4;

%f3:Correlation
f3_0 = 0;
f3_45 = 0;
f3_90 = 0;
f3_135 = 0;
for i = 1:Ng
    for j = 1:Ng
        f3_0 = f3_0 + (((i*j)*gclm_0_norm(i,j))-(mux_0*muy_0))/(stdx_0*stdy_0);
        f3_45 = f3_45 + (((i*j)*gclm_45_norm(i,j))-(mux_45*muy_45))/(stdx_45*stdy_45);
        f3_90 = f3_90 + (((i*j)*gclm_90_norm(i,j))-(mux_90*muy_90))/(stdx_90*stdy_90);
        f3_135 = f3_135 + (((i*j)*gclm_135_norm(i,j))-(mux_135*muy_135))/(stdx_135*stdy_135);
    end
end
f3 = (f3_0 + f3_45 + f3_90 + f3_135)/4;

%f5:Inverse Difference Moment
f5_0 = 0;
f5_45 = 0;
f5_90 = 0;
f5_135 = 0;
for i = 1:Ng
    for j = 1:Ng
        f5_0 = f5_0 + (gclm_0_norm(i,j))/(1+((i-j).^2));
        f5_45 = f5_45 + (gclm_45_norm(i,j))/(1+((i-j).^2));
        f5_90 = f5_90 + (gclm_90_norm(i,j))/(1+((i-j).^2));
        f5_135 = f5_135 + (gclm_135_norm(i,j))/(1+((i-j).^2));
    end
end
f5 = (f5_0 + f5_45 + f5_90 + f5_135)/4;

%f6:Sum Average
f6_0 = 0;
f6_45 = 0;
f6_90 = 0;
f6_135 = 0;
for i = 2:2*Ng
    f6_0 = f6_0 + (i*pxpy_0(i));
    f6_45 = f6_45 + (i*pxpy_45(i));
    f6_90 = f6_90 + (i*pxpy_90(i));
    f6_135 = f6_135 + (i*pxpy_135(i));
end
f6 = (f6_0 + f6_45 + f6_90 + f6_135)/4;


%f8:Sum Entropy
f8_0 = 0;
f8_45 = 0;
f8_90 = 0;
f8_135 = 0;
for i=2:2*Ng
     f8_0 = (f8_0 + ((pxpy_0(i)) * log(1+pxpy_0(i))));
     f8_45 = (f8_45 + ((pxpy_45(i)) * log(1+pxpy_45(i))));
     f8_90 = (f8_90 + ((pxpy_90(i)) * log(1+pxpy_90(i))));
     f8_135 = (f8_135 + ((pxpy_135(i)) * log(1+pxpy_135(i))));
end
f8 = -(f8_0 + f8_45 + f8_90 + f8_135)/4;

%f7:Sum Variance
f7_0 = 0;
f7_45 = 0;
f7_90 = 0;
f7_135 = 0;
for i = 2:2*Ng
    f7_0 = f7_0 + (((i-f8_0).^2)*pxpy_0(i));
    f7_45 = f7_45 + (((i-f8_45).^2)*pxpy_45(i));
    f7_90 = f7_90 + (((i-f8_90).^2)*pxpy_90(i));
    f7_135 = f7_135 + (((i-f8_135).^2)*pxpy_135(i));
end
f7 = (f7_0 + f7_45 + f7_90 + f7_135)/4;

%f9:Entropy
f9_0 = 0;
f9_45 = 0;
f9_90 = 0;
f9_135 = 0;
for i = 1:Ng
    for j = 1:Ng
        f9_0 = (f9_0 +((gclm_0_norm(i,j))*log(1+gclm_0_norm(i,j))));
        f9_45 = (f9_45 +((gclm_45_norm(i,j))*log(1+gclm_45_norm(i,j))));
        f9_90 = (f9_90 +((gclm_90_norm(i,j))*log(1+gclm_90_norm(i,j))));
        f9_135 = (f9_135 +((gclm_135_norm(i,j))*log(1+gclm_135_norm(i,j))));
    end
end
f9 = -(f9_0 + f9_45 + f9_90 + f9_135)/4;

return
end
