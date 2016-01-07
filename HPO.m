%INPUT AND OUTPUT FILE PREPARATION
%load input file
prompt = 'Input file name: ';
str = input(prompt,'s');
in_file=load(str);



%choose left or right
prompt = 'Extract left or right wrist: 1 (for left), 2 (for right): ';
st = input(prompt,'s');


%OUTPUT FILE
%output file is in ".-arff" file format in order to use WEKA tool later
%Weka link: http://www.cs.waikato.ac.nz/ml/weka/
%column 23, 24, 25 in input file is left wrist sensor's acceleration
%column 32, 33, 34 in input file is right wrist sensor's acceleration
%load acceleration data input to x, y, z (x acceleration, y acceleration, z
%accleration)
if st=='1'
    x = in_file(:,23);
    y = in_file(:,24);
    z = in_file(:,25);
    out_file = strcat(str(:,1:(size(str,2)-4)), '-left.arff');
else
    x = in_file(:,32);
    y = in_file(:,33);
    z = in_file(:,34);
    out_file = strcat(str(:,1:(size(str,2)-4)), '-right.arff');
end



%load blank file to output file (in case file existed data)
blank_file = [];            
dlmwrite(out_file, blank_file);



% load additional data: timestamp and activity label
% column 1, and 115 in input file are timestamp and activity label
% data matrix in format of
% [timestamp x-acceleration y-acceleration z-acceleration activityLabel]
time_stamp = in_file(:,1);  %time stamp
lbl = in_file(:,115);       %label of activity
ft_mat_lbl = [time_stamp x y z lbl]; %data matrix


%PREPARE OUTPUT ARFF FORMAT FILE
fid = fopen(out_file,'w'); % Open output file
if st=='1'
    relation = '@RELATION LeftWrist';
else
    relation = '@RELATION RightWrist';
end
if fid ~= -1
    fprintf(fid,'%s\r\n',relation);       %# Write the relation
    %12 attribute/features
    fprintf(fid,'%s\r\n', '@ATTRIBUTE mean_x numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE mean_y numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE mean_z numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE energy_x numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE energy_y numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE energy_z numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE normEn_x numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE normEn_y numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE normEn_z numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE corr_x numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE corr_y numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE corr_z numeric');
    fprintf(fid,'%s\r\n', '@ATTRIBUTE label{0.000000,101.000000,102.000000,104.000000,105.000000}');
    fprintf(fid, '%s\r\n', '@DATA');
    fclose(fid);                     %# Close output file
end

%PREPROCESSING INPUT
% remove noise data - NaN
ft_mat_lbl(any(isnan(ft_mat_lbl),2),:)=[];
dlmwrite('data_raw', ft_mat_lbl,'precision','%.6f');


%WINDOW SIZE AND WINDOW SLIDE DESCRIPTION
%my_window size
%window size = 128
%window slide 50% (window size/2)
window_size = 128;
window_start = 1;
window_end = window_start + window_size - 1;
window_slide = window_size/2;

%FEATURE EXTRACTION
%each window extracts one segment
%segment is one line of output data
%one line of output data contains 12 features, which are in the order: 
%mean of x acceleration, mean of y acceleration, mean of z acceleration, energy in x axis,
%energy in y axis, energy in z axis, frequency domain entropy (x axis), frequency
%domain entropy (y axis), frequency domain entropy (z axis), correlation between x-y axis,
%correlation between y-z axis, correlation between x-z axis.

i = 1;
segment_lbl = ft_mat_lbl(1,5); %first segment's activity label
segment = [];

while i<=size(ft_mat_lbl,1)
    point_lbl = ft_mat_lbl(i,5);
    if point_lbl == segment_lbl
        segment = [segment; ft_mat_lbl(i,:)];
        
    else % start extract
        window_start = 1;
        window_end = window_start + window_size - 1;
        
        % each window should capture around 5 second activity movement
        % in this case, sampling rate is 30Hz, window size is 128
        % Hence, the duration for each window is around 4.2 seconds
        if window_end >= size(segment,1) && size(segment,1) > 1% only 1 window
            if segment(size(segment,1),1) - segment(1,1) <=5000 %5s
                wind = segment(:,2:4);
                %3 MEAN FEATURES IN X-AXIS, Y-AXIS, Z-AXIS
                mean_ft = mean(wind);
                
                %3 ENGERGY FEATURES IN X-AXIS, Y-AXIS, Z-AXIS
                fft_x = abs(fft(wind(:,1)));
                fft_x(1,:)=[];
                fft_y = abs(fft(wind(:,2)));
                fft_y(1,:)=[];
                fft_z = abs(fft(wind(:,3)));
                fft_z(1,:)=[];
                
                energy_x = sum(fft_x.^2)/size(fft_x, 1);
                energy_y = sum(fft_y.^2)/size(fft_y, 1);
                energy_z = sum(fft_z.^2)/size(fft_z, 1);
                
                %3 FREQUENCY DOMAIN ENTROPY FEATURES IN X-AXIS, Y-AXIS, Z-AXIS
                fft_x = fft_x(fft_x ~= 0);
                fft_y = fft_y(fft_y ~= 0);
                fft_z = fft_z(fft_z ~= 0);
                norm_x = fft_x/sum(fft_x);
                norm_y = fft_y/sum(fft_y);
                norm_z = fft_z/sum(fft_z);
                en_x = -sum(norm_x.*log2(norm_x));
                en_y = -sum(norm_y.*log2(norm_y));
                en_z = -sum(norm_z.*log2(norm_z));
                
                %3 CORRELATION BETWEEN X-Y, Y-Z, X-Z
                corr_xy = corr(wind(:,1), wind(:,2));
                corr_yz = corr(wind(:,2), wind(:,3));
                corr_xz = corr(wind(:,1), wind(:,3));
                corr_xy(isnan(corr_xy))=0;
                corr_yz(isnan(corr_yz))=0;
                corr_xz(isnan(corr_xz))=0;
                
                % write output
                final_ft = [mean_ft energy_x energy_y energy_z en_x en_y en_z corr_xy corr_yz corr_xz segment_lbl];
                %final_ft(any(isnan(final_ft),2),:)=[];
                dlmwrite(out_file, final_ft,'-append','precision','%.6f');
            end
        else %multiple windows
            checked_end = 0;
            while window_start<size(segment,1) && window_end<size(segment,1) && window_start < window_end
                wind_temp = segment(window_start:window_end,:);
                if segment(window_end,1) - segment(window_start,1) <= 5000
                    wind = wind_temp(:, 2:4);
                    mean_ft = mean(wind);

                    fft_x = abs(fft(wind(:,1)));
                    fft_x(1,:)=[];
                    fft_y = abs(fft(wind(:,2)));
                    fft_y(1,:)=[];
                    fft_z = abs(fft(wind(:,3)));
                    fft_z(1,:)=[];

                    energy_x = sum(fft_x.^2)/size(fft_x, 1);
                    energy_y = sum(fft_y.^2)/size(fft_y, 1);
                    energy_z = sum(fft_z.^2)/size(fft_z, 1);
                    
                    fft_x = fft_x(fft_x ~= 0);
                    fft_y = fft_y(fft_y ~= 0);
                    fft_z = fft_z(fft_z ~= 0);
                    norm_x = fft_x/sum(fft_x);
                    norm_y = fft_y/sum(fft_y);
                    norm_z = fft_z/sum(fft_z);
                    
                    en_x = -sum(norm_x.*log2(norm_x));
                    en_y = -sum(norm_y.*log2(norm_y));
                    en_z = -sum(norm_z.*log2(norm_z));

                    corr_xy = corr(wind(:,1), wind(:,2));
                    corr_yz = corr(wind(:,2), wind(:,3));
                    corr_xz = corr(wind(:,1), wind(:,3));
                    corr_xy(isnan(corr_xy))=0;
                    corr_yz(isnan(corr_yz))=0;
                    corr_xz(isnan(corr_xz))=0;
                    
                    final_ft = [mean_ft energy_x energy_y energy_z en_x en_y en_z corr_xy corr_yz corr_xz segment_lbl];
                    %final_ft(any(isnan(final_ft),2),:)=[];
                    dlmwrite(out_file, final_ft,'-append','precision','%.6f');
                end
                window_start = window_start + window_slide;
                window_end = window_end + window_slide;
                if window_end >= size(segment,1) && checked_end == 0
                    window_end = size(segment,1);
                    checked_end = 1;
                end
            end
        end
        segment_lbl = point_lbl;
        segment = ft_mat_lbl(i,:);
    end
    
    %final segment
    if i == size(ft_mat_lbl,1)
        window_start = 1;
        window_end = window_start + window_size - 1;
        
        if window_end >= size(segment,1) && size(segment,1)>1 % only 1 window
            if segment(size(segment,1),1) - segment(1,1) <=5000 %5s
                wind = segment(:,2:4);
                mean_ft = mean(wind);
                fft_x = abs(fft(wind(:,1)));
                fft_x(1,:)=[];
                fft_y = abs(fft(wind(:,2)));
                fft_y(1,:)=[];
                fft_z = abs(fft(wind(:,3)));
                fft_z(1,:)=[];
                
                energy_x = sum(fft_x.^2)/size(fft_x, 1);
                energy_y = sum(fft_y.^2)/size(fft_y, 1);
                energy_z = sum(fft_z.^2)/size(fft_z, 1);
                
                fft_x = fft_x(fft_x ~= 0);
                fft_y = fft_y(fft_y ~= 0);
                fft_z = fft_z(fft_z ~= 0);
                norm_x = fft_x/sum(fft_x);
                norm_y = fft_y/sum(fft_y);
                norm_z = fft_z/sum(fft_z);
                
                en_x = -sum(norm_x.*log2(norm_x));
                en_y = -sum(norm_y.*log2(norm_y));
                en_z = -sum(norm_z.*log2(norm_z));
                
                corr_xy = corr(wind(:,1), wind(:,2));
                corr_yz = corr(wind(:,2), wind(:,3));
                corr_xz = corr(wind(:,1), wind(:,3));
                corr_xy(isnan(corr_xy))=0;
                corr_yz(isnan(corr_yz))=0;
                corr_xz(isnan(corr_xz))=0;
                
                final_ft = [mean_ft energy_x energy_y energy_z en_x en_y en_z corr_xy corr_yz corr_xz segment_lbl];
                %final_ft(any(isnan(final_ft),2),:)=[];
                dlmwrite(out_file, final_ft,'-append','precision','%.6f');
            end
        else %multiple windows
            checked_end = 0;
            while window_start<size(segment,1) && window_end<size(segment,1) && window_start < window_end
                wind_temp = segment(window_start:window_end,:);
                if segment(window_end,1) - segment(window_start,1) <= 5000
                    wind = wind_temp(:, 2:4);
                    mean_ft = mean(wind);

                    fft_x = abs(fft(wind(:,1)));
                    fft_x(1,:)=[];
                    fft_y = abs(fft(wind(:,2)));
                    fft_y(1,:)=[];
                    fft_z = abs(fft(wind(:,3)));
                    fft_z(1,:)=[];

                    energy_x = sum(fft_x.^2)/size(fft_x, 1);
                    energy_y = sum(fft_y.^2)/size(fft_y, 1);
                    energy_z = sum(fft_z.^2)/size(fft_z, 1);

                    fft_x = fft_x(fft_x ~= 0);
                    fft_y = fft_y(fft_y ~= 0);
                    fft_z = fft_z(fft_z ~= 0);
                    norm_x = fft_x/sum(fft_x);
                    norm_y = fft_y/sum(fft_y);
                    norm_z = fft_z/sum(fft_z);
                    
                    en_x = -sum(norm_x.*log2(norm_x));
                    en_y = -sum(norm_y.*log2(norm_y));
                    en_z = -sum(norm_z.*log2(norm_z));

                    corr_xy = corr(wind(:,1), wind(:,2));
                    corr_yz = corr(wind(:,2), wind(:,3));
                    corr_xz = corr(wind(:,1), wind(:,3));
                    corr_xy(isnan(corr_xy))=0;
                    corr_yz(isnan(corr_yz))=0;
                    corr_xz(isnan(corr_xz))=0;

                    final_ft = [mean_ft energy_x energy_y energy_z en_x en_y en_z corr_xy corr_yz corr_xz segment_lbl];
                    %final_ft(any(isnan(final_ft),2),:)=[];
                    dlmwrite(out_file, final_ft,'-append','precision','%.6f');
                end
                window_start = window_start + window_slide;
                window_end = window_end + window_slide;
                if window_end >= size(segment,1) && checked_end == 0
                    window_end = size(segment,1);
                    checked_end = 1;
                end
            end
        end
    end
    i = i+1;
end %end while