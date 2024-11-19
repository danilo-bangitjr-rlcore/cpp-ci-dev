SELECT
   time_bucket(INTERVAL '0:05:00', time) + '0:05:00' as time_bucket,
   name,
   last((fields->'val')::float AS val, time)
FROM sensors
WHERE 
	time > TIMESTAMP '2024-11-13T12:08:30+00:00'
	AND time < TIMESTAMP '2024-11-13T18:08:30+00:00'
	AND (
		name = 'AI0879C'
	)
GROUP BY time_bucket, name
ORDER BY time_bucket ASC, name ASC;


SELECT
	time_bucket(INTERVAL '0:00:30', time) + '0:00:30' as time_bucket,
	name,
	last((fields->'val')::float, time) AS val
FROM scrubber4
WHERE	
	time > TIMESTAMP '2024-11-13T18:05:30+00:00'
	AND time < TIMESTAMP '2024-11-13T18:08:30+00:00'	
	AND name = 'AI0879C'
GROUP BY time_bucket, name
ORDER BY time_bucket ASC, name ASC;

select time, name, (fields->'val')::float as val from scrubber4
WHERE
time > TIMESTAMP '2024-11-13T18:00:30+00:00'
AND time < TIMESTAMP '2024-11-13T18:08:30+00:00'
AND name = 'AI0879C'
ORDER BY time ASC;

select time, name, (fields->'val')::float as val from scrubber4
WHERE
time > TIMESTAMP '2024-11-13T18:00:30+00:00'
AND time < TIMESTAMP '2024-11-13T18:08:30+00:00'
AND name = 'AI0879C'
ORDER BY time ASC;

select time, name, (fields->'val')::float as val 
FROM sensors
WHERE
time > TIMESTAMP '2024-11-12 22:57:13+00'
AND time < TIMESTAMP '2024-11-12 22:59:13+00'
AND name = 'sensor1'
ORDER BY time ASC;


SELECT
	time_bucket(INTERVAL '0:00:15', time) + '0:00:15' as time_bucket,
	name,
	last((fields->'val')::float, time) AS val
FROM sensors
WHERE	
	time > TIMESTAMP '2024-11-12 22:57:13+00'
	AND time < TIMESTAMP '2024-11-12 22:59:13+00'
	AND name = 'sensor1'
GROUP BY time_bucket, name
ORDER BY time_bucket ASC, name ASC;

SELECT
	time_bucket(INTERVAL '0:00:15', time) + '0:00:15' as time_bucket,
	name,
	avg((fields->'val')::float) AS val
FROM sensors
WHERE	
	time > TIMESTAMP '2024-11-12 22:57:13+00'
	AND time < TIMESTAMP '2024-11-12 22:59:13+00'
	AND name = 'sensor1'
GROUP BY time_bucket, name
ORDER BY time_bucket ASC, name ASC;


SELECT
	time_bucket(INTERVAL '0:00:15', time, origin => TIMESTAMP '2024-11-12 22:58:45+00') + '0:00:15' as time_bucket,
	name,
	avg((fields->'val')::float) AS val
FROM sensors
WHERE	
	time > TIMESTAMP '2024-11-12 22:58:45+00'
	AND time < TIMESTAMP '2024-11-12 22:59:00+00'
	AND name = 'sensor1'
GROUP BY time_bucket, name
ORDER BY time_bucket ASC, name ASC;
