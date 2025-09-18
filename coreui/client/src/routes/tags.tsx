import { createFileRoute } from '@tanstack/react-router';
import { ObsCard } from '../components/obs-card-components/ObsCard';
import { TrashIcon } from '../components/icons/TrashIcon';
import { useTagConfigs } from '../utils/useTagConfigs';

export const Route = createFileRoute('/tags')({
  component: Tags,
});

function Tags() {
  const {
    configs,
    isPending,
    error,
    handleConfigChange,
    updateTag,
    addObsCard,
    deleteTag,
  } = useTagConfigs();

  if (isPending) return 'Loading tags..';
  if (error) return ' :-( ' + error.message;

  return (
    <div className="p-2 space-y-6 flex flex-col gap-1 justify-center">
      {configs.map((config, index) => (
        <div key={index} className="relative">
          <ObsCard
            config={config}
            onConfigChange={(updatedConfig) =>
              handleConfigChange(index, updatedConfig)
            }
            onSave={
              config.modified ? () => updateTag(config, index) : undefined
            }
          />
          <button
            onClick={() => deleteTag(config.name, index)}
            className="absolute top-0 right-0 p-2 bg-transparent hover:bg-gray-200 rounded"
            aria-label="Delete"
            type="button"
          >
            <TrashIcon className="w-5 h-5 text-gray-500" />
          </button>
        </div>
      ))}
      <button
        onClick={addObsCard}
        className="w-full self-start mb-2 px-4 py-2 rounded bg-gray-200 text-gray-800 border border-gray-300 hover:bg-gray-300 transition"
      >
        Add Obs Card
      </button>
    </div>
  );
}
