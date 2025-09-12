import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { ObsCard } from '../components/obs-card-components/ObsCard';
import { API_ENDPOINTS, get } from '../utils/api';

export const Route = createFileRoute('/tags')({
  component: Tags,
});

type RawTag = {
  name?: string;
  connection_id?: string;
  operating_range?: [number, number];
  expected_range?: [number, number];
  is_computed?: boolean;
};

function Tags() {
  const { isPending, error, data } = useQuery({
    queryKey: ['tags'],
    queryFn: async () => {
      const response = await get(API_ENDPOINTS.configs.tags);
      if (!response.ok) throw new Error('Failed to fetch tags');
      return response.json();
    },
  });

  if (isPending) return 'Loading tags..';
  if (error) return ' :-( ' + error.message;

  // Filter and map tags for ObsCard
  const filtered = (data.tags as RawTag[]).filter(
    (t) =>
      t.name &&
      (t.connection_id || t.operating_range || t.expected_range) &&
      !t.is_computed
  );

  const obsConfigs = filtered.map((tag) => ({
    name: tag.name!,
    connection_id: tag.connection_id,
    tags: [
      ...(tag.operating_range
        ? [
            {
              label: 'Operating',
              values: tag.operating_range as [number, number],
            },
          ]
        : []),
      ...(tag.expected_range
        ? [
            {
              label: 'Expected',
              values: tag.expected_range as [number, number],
            },
          ]
        : []),
    ],
  }));

  return (
    <div className="p-2 space-y-6 flex flex-col gap-1 justify-center">
      {obsConfigs.map((config) => (
        <ObsCard key={config.name} config={config} />
      ))}
    </div>
  );
}
