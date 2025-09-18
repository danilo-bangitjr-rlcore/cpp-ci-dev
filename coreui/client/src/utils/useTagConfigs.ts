import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { API_ENDPOINTS, get } from './api';
import { buildObsConfig, createEmptyObsConfig } from '../types/tag-types';
import type { RawTag, ObsCardConfig } from '../types/tag-types';

const TEMP_CONFIG_NAME = 'main_backwash';

function filterAndMapTags(tags: RawTag[]): ObsCardConfig[] {
  return tags
    .map((tag, originalIndex) => ({ tag, originalIndex }))
    .filter(
      ({ tag }) =>
        (tag.name || tag.name === '') &&
        (tag.connection_id || tag.operating_range || tag.expected_range) &&
        !tag.is_computed
    )
    .map(({ tag, originalIndex }) => buildObsConfig(tag, originalIndex));
}

export function useTagConfigs() {
  const [configs, setConfigs] = useState<ObsCardConfig[]>([]);
  const queryClient = useQueryClient();

  const { isPending, error, data } = useQuery({
    queryKey: ['tags'],
    queryFn: async () => {
      const response = await get(API_ENDPOINTS.configs.tags);
      if (!response.ok) throw new Error('Failed to fetch tags');
      return response.json();
    },
  });

  useEffect(() => {
    if (data?.tags) {
      setConfigs(filterAndMapTags(data.tags as RawTag[]));
    }
  }, [data]);

  const getIndexForTag = (tagName: string): number => {
    return data?.tags?.findIndex((t: RawTag) => t.name === tagName) ?? -1;
  };

  const handleConfigChange = (index: number, updatedConfig: ObsCardConfig) => {
    setConfigs((prev) =>
      prev.map((config, i) =>
        i === index ? { ...updatedConfig, modified: true } : config
      )
    );
  };

  const deleteConfig = async (
    configName: string,
    index: number
  ): Promise<void> => {
    try {
      const response = await fetch(
        API_ENDPOINTS.configs.delete_raw_tag(configName, index),
        {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config_name: configName, index }),
        }
      );
      await response.json();
      queryClient.invalidateQueries({ queryKey: ['tags'] });
    } catch (error) {
      console.error('Error deleting tag:', error);
    }
  };

  const addTag = async (newTag: ObsCardConfig) => {
    const rawTag = {
      tag: {
        name: newTag.name,
        connection_id: newTag.connection_id,
        operating_range: newTag.tags[0].operating_range,
        expected_range: newTag.tags[0].expected_range,
        yellow_bounds: newTag.tags[0].yellow_bounds,
        red_bounds: newTag.tags[0].red_bounds,
      },
    };
    try {
      const response = await fetch(
        API_ENDPOINTS.configs.add_raw_tag(TEMP_CONFIG_NAME),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(rawTag),
        }
      );
      if (!response.ok) throw new Error('Failed to add tag');
      await response.json();
      queryClient.invalidateQueries({ queryKey: ['tags'] });
    } catch (error) {
      console.error('Error adding tag:', error);
    }
  };

  const updateTag = async (config: ObsCardConfig, tagIndex: number) => {
    const rawTag = {
      tag: {
        name: config.name,
        connection_id: config.connection_id,
        operating_range: config.tags[0].operating_range,
        expected_range: config.tags[0].expected_range,
        yellow_bounds: config.tags[0].yellow_bounds,
        red_bounds: config.tags[0].red_bounds,
      },
    };
    try {
      const response = await fetch(
        API_ENDPOINTS.configs.update_raw_tag(
          TEMP_CONFIG_NAME,
          config.originalIndex
        ),
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(rawTag),
        }
      );
      if (!response.ok) throw new Error('Failed to update tag');
      await response.json();
      setConfigs((prev) =>
        prev.map((c, i) => (i === tagIndex ? { ...c, modified: false } : c))
      );
      queryClient.invalidateQueries({ queryKey: ['tags'] });
    } catch (error) {
      console.error('Error updating tag:', error);
    }
  };

  const addObsCard = () => {
    const emptyTag = createEmptyObsConfig(data?.tags?.length ?? 0);
    setConfigs((prev) => [...prev, emptyTag]);
    addTag(emptyTag);
  };

  const deleteTag = (tagName: string, index: number) => {
    setConfigs((prev) => prev.filter((_, i) => i !== index));
    const tagIndex = getIndexForTag(tagName);
    deleteConfig(TEMP_CONFIG_NAME, tagIndex);
  };

  return {
    configs,
    isPending,
    error,
    handleConfigChange,
    updateTag,
    addObsCard,
    deleteTag,
  };
}
