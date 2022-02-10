function ap = score_ap_from_ranks (ranks, returned, nres)

    isPos = zeros(returned,1); % set the list of 0s for all position
    isPos(ranks) = 1; % set the position of true as 1
    prec = cumsum(isPos)./[1:returned]'; % 
    recall = cumsum(isPos)/nres; % 
    ap = diff([0;recall]')*prec;        
end