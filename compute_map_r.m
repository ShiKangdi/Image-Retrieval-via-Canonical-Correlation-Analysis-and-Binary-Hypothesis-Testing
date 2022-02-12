function [oldmap, newmap, ns, aps] = compute_map_r (ranks, gnd, verbose)
% ranks: contain the position of true, because the number of truth is
% different for different query image, so we pad zero to make sure the
% dimension of rank matrix is not a problem
% each item is a index of image.
% gnd: is the ground truth.

if nargin < 3
    verbose = false; % if # of input is smaller than 3, verbose is false
end

oldmap = 0;
newmap = 0;
nq = numel (gnd);  % number of queries,so gnd is the list of query image

aps = zeros (nq, 1); % set the list for ap of each query image
ns = 0; % ?

for i = 1:nq % starting from 1st query image to the last
    %qgnd = gnd(i).hard; % get the index of ok label for ith query image
    %qgnd = [qgnd, gnd(i).easy];
    qgnd = gnd(i).ok;
    if isfield (gnd(i), 'junk') % if the ith query contain junk label or not
        qgndj = gnd(i).junk; % if yes, get the index of junk image into list qgndj
    else
        qgndj = []; % if not, set list qgndj empty
    end
     
    % positions of positive and junk images
    [~, pos] = intersect (ranks (:,i), qgnd); % get intersection index between ith column in rank array and ok label array
    [~, junk] = intersect (ranks (:,i), qgndj); % get intersection index between ith column in rank array and junk label array
    
    pos = sort(pos); % sort the index of pos
    junk = sort(junk); % sort the index of junk
    
    k = 0;
    ij = 1;
    
    if length (junk)
        % decrease positions of positives based on the number of junk images appearing before them
        ip = 1;
        while ip <= numel (pos)
            
            while ( ij <= length (junk) && pos (ip) > junk (ij) )
                k = k + 1;
                ij = ij + 1;
            end
            
            pos (ip) = pos (ip) - k;
            ip = ip + 1;
        end
    end
    
    newap = score_ap_from_ranks (pos, size(ranks,1), length (qgnd));
    
    isPos = zeros(size(ranks,1),1);
    isPos(pos) = 1;
    recall_4 = sum(isPos(1:4))/length (qgnd);
    
    if verbose
        fprintf ('query no %d -> gnd = ', i);
        fprintf ('%d ', qgnd);
        fprintf ('\n              tp ranks = ');
        fprintf ('%d ', pos);
        fprintf (' -> ap=%.3f\n', newap);
    end
    newmap = newmap + newap;
    aps(i) = newap;
    ns = ns + 4*recall_4;
end
newmap = newmap / nq;
ns = ns / nq;
end


