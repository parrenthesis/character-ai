#!/usr/bin/env python3
"""
Convert old character YAML format to new schema format.
This script migrates characters from the old format to the new schema-based format.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List

def convert_old_character(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old character format to new schema format."""
    
    # Extract basic info
    name = old_data.get('name', 'Unknown')
    language = old_data.get('language', 'en')
    voice_style = old_data.get('voice_style', 'friendly')
    
    # Extract dimensions
    dimensions = old_data.get('dimensions', {})
    personality_traits = dimensions.get('personality_traits', [])
    topics = dimensions.get('topics', [])
    abilities = dimensions.get('abilities', [])
    archetype = dimensions.get('archetype', 'friend')
    backstory = dimensions.get('backstory', '')
    species = dimensions.get('species', 'human')
    
    # Create new schema format
    new_profile = {
        'schema_version': 1,
        'id': name.lower().replace(' ', '_'),
        'display_name': name,
        'character_type': species,
        'language': language,
        'traits': {
            'personality': ', '.join(personality_traits) if personality_traits else 'friendly',
            'speech_pattern': voice_style,
            'catchphrase': f"I am {name}",
            'special_abilities': ', '.join(abilities) if abilities else 'none',
            'voice_style': voice_style
        },
        'topics': topics,
        'safety': {
            'content_filter': True,
            'age_appropriate': True,
            'banned_topics': ['violence', 'inappropriate content']
        },
        'llm': {
            'model': 'phi-3-mini-4k-instruct',
            'temperature': 0.7,
            'max_tokens': 200
        },
        'stt': {
            'model': 'whisper-base',
            'language': language
        },
        'tts': {
            'model': 'xtts',
            'voice_id': f"{name.lower().replace(' ', '_')}_voice",
            'speed': 1.0
        },
        'consent': {
            'subject': 'adult_guardian',
            'date': '2024-01-15',
            'purpose': 'educational_entertainment',
            'retention': '1_year'
        }
    }
    
    return new_profile

def create_prompts_file(character_name: str, backstory: str, personality_traits: List[str]) -> Dict[str, Any]:
    """Create prompts.yaml file for a character."""
    
    # Create system prompt based on character info
    system_prompt = f"""You are {character_name}, a {', '.join(personality_traits) if personality_traits else 'friendly'} character.
{backstory}

Key characteristics:
- {', '.join(personality_traits) if personality_traits else 'Friendly and helpful'}
- Speaks in a {', '.join(personality_traits) if personality_traits else 'friendly'} manner
- Always helpful and encouraging

Always respond as {character_name} would, maintaining your character and personality."""

    safety_prompt = """Remember that you are an educational character for children. Keep responses
appropriate for all ages and focus on positive themes like friendship,
exploration, and learning."""

    conversation_starters = [
        f"Hello! I'm {character_name}. How can I help you today?",
        f"Greetings! I'm {character_name}. What would you like to know?",
        f"Hi there! I'm {character_name}. What can we explore together?"
    ]
    
    return {
        'system_prompt': system_prompt,
        'safety_prompt': safety_prompt,
        'conversation_starters': conversation_starters
    }

def main():
    """Convert all old character files to new format."""
    
    # Paths
    old_characters_dir = Path('characters')
    new_characters_dir = Path('configs/characters')
    
    # Create new directory structure
    new_characters_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each old character file
    for old_file in old_characters_dir.glob('*.yaml'):
        print(f"Converting {old_file.name}...")
        
        # Read old format
        with open(old_file, 'r') as f:
            old_data = yaml.safe_load(f)
        
        # Convert to new format
        new_profile = convert_old_character(old_data)
        
        # Create character directory
        character_id = new_profile['id']
        character_dir = new_characters_dir / character_id
        character_dir.mkdir(exist_ok=True)
        
        # Write profile.yaml
        profile_file = character_dir / 'profile.yaml'
        with open(profile_file, 'w') as f:
            yaml.dump(new_profile, f, default_flow_style=False, sort_keys=False)
        
        # Create prompts.yaml
        dimensions = old_data.get('dimensions', {})
        personality_traits = dimensions.get('personality_traits', [])
        backstory = dimensions.get('backstory', '')
        
        prompts_data = create_prompts_file(character_id, backstory, personality_traits)
        prompts_file = character_dir / 'prompts.yaml'
        with open(prompts_file, 'w') as f:
            yaml.dump(prompts_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Created {character_dir}/profile.yaml")
        print(f"  ✓ Created {character_dir}/prompts.yaml")
    
    print(f"\n✅ Conversion complete! Converted {len(list(old_characters_dir.glob('*.yaml')))} characters.")
    print(f"📁 New characters are in: {new_characters_dir}")
    print(f"🗑️  Old characters directory can be removed: {old_characters_dir}")

if __name__ == '__main__':
    main()
