# Hierarchy for suncg & training data generation for grass code

The components are:

- room_select: pick up suncg rooms based on room type
- train_test_split: get train & test split by cell2txt
- stats2hie_batch: generate hierarchy for suncg rooms
- visulization: visualize room hierarchy (should have object images)
- grass_data_gen: generate training data for grass code

1. room_select
```python
cd room_select
python scn2room_suncg_*.py
```

2. stats2hie_batch
```python
cd stats2hie_batch
python stats2hie_batch.py
```

3. visulization
```python
cd visulization
python tree_vis.py
```

4. grass_data_gen
```python
python grass_data_gen.py
```
