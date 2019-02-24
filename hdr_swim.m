data = {
    importdata('10k/normalized.csv', ',', 1);
    importdata('51k/normalized.csv', ',', 1);
    importdata('120k/normalized.csv', ',', 1);
};

ds = [];
uniques = cell(length(data), 1);
for i = 1:length(data)
    [U, idx] = unique(data{i}.data(:,1));
    ds((length(ds)+1):(length(ds)+size(U, 1)),1) = U;
    uniques{i} = idx;
end

re_ = NaN(size(ds, 1), length(data) * 2);
im_ = NaN(size(ds, 1), length(data) * 2);
for i = 1:length(data)
    D = data{i}.data(uniques{i},:);
    re_(:,(i*2-1):(i*2)) = interp1(D(:,1), D(:,2:3), ds);
    im_(:,(i*2-1):(i*2)) = interp1(D(:,1), D(:,4:5), ds);
end

re = sum(re_(:,2:2:end) .* re_(:,1:2:end), 2) ./ sum(re_(:,2:2:end), 2);
im = sum(im_(:,2:2:end) .* im_(:,1:2:end), 2) ./ sum(im_(:,2:2:end), 2);

[d_sorted, order] = sort(ds);

subplot(2, 1, 1);
plot(d_sorted, re(order));
subplot(2, 1, 2);
plot(d_sorted, im(order));

csvwrite('re.csv', [d_sorted, re(order)]);
csvwrite('im.csv', [d_sorted, im(order)]);