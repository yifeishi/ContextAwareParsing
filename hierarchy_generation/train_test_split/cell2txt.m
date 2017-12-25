clear all
trainingData = load('SUNCG_test_sceneId');
fid = fopen('SUNCG_test_sceneId.txt','wt');
for i = 1:length(trainingData.allSceneId)
    fprintf(fid,'%s\n',trainingData.allSceneId{i});
end
fclose(fid);