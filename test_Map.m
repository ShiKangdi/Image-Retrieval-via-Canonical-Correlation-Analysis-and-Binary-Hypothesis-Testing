load('gnd_oxford5k.mat')
%load('gnd_rparis6k.mat')
%load('gnd_paris6k.mat')
%load('gnd_oxford5k.mat')

Folder = '' % Place Your Folder address in here. The folder stores the results in txt format. The name of each .txt file is "query0.txt","query1.txt","query2.txt"....
dim = [512,450,400,300,200,100,50,25];
for n = 1:length(dim)
    for m = 0:54
        fid = fopen(strcat(Folder + '/query',string(m),'.txt'));
        data = textscan(fid,'%s'); 
        fclose(fid);
        for j = 1:length(imlist)
            Index = find(contains(imlist,data{1}{j}));
            ranks(j,m+1)= Index;
        end
    end
    [oldmap, newmap, ns, aps] = compute_map_r(ranks, gnd);
    fprintf(strcat(a,b,c, string(dim(n)),'/', string(newaps)))
    fprintf('\n')
end
