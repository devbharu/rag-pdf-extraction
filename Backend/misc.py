import trafilatura
from trafilatura import extract
import json

# Fetch the URL
url = 'https://docs.exa.ai/integrations/crew-ai-docs'
print(f"Fetching URL: {url}\n")
print("=" * 80)

downloaded = trafilatura.fetch_url(url)

if downloaded:
    # Extract with all available options enabled
    result = extract(
        downloaded,
        include_comments=True,      # Include comments
        include_tables=True,        # Include tables
        include_images=True,        # Include image information
        include_links=True,         # Include links
        output_format='json',       # Get structured JSON output
        with_metadata=True,         # Include metadata
        url=url                     # Pass URL for better metadata
    )
    
    if result:
        # Parse JSON result
        data = json.loads(result)
        
        # Print metadata
        print("\n📋 METADATA:")
        print("-" * 80)
        if 'title' in data:
            print(f"Title: {data['title']}")
        if 'author' in data:
            print(f"Author: {data['author']}")
        if 'date' in data:
            print(f"Date: {data['date']}")
        if 'url' in data:
            print(f"URL: {data['url']}")
        if 'hostname' in data:
            print(f"Hostname: {data['hostname']}")
        if 'description' in data:
            print(f"Description: {data['description']}")
        if 'sitename' in data:
            print(f"Site Name: {data['sitename']}")
        if 'categories' in data:
            print(f"Categories: {data['categories']}")
        if 'tags' in data:
            print(f"Tags: {data['tags']}")
        
        # Print main text content
        print("\n📝 MAIN CONTENT:")
        print("-" * 80)
        if 'text' in data:
            print(data['text'])
        
        # Print comments if available
        if 'comments' in data and data['comments']:
            print("\n💬 COMMENTS:")
            print("-" * 80)
            print(data['comments'])
        
        # Print links if available
        if 'links' in data and data['links']:
            print("\n🔗 LINKS:")
            print("-" * 80)
            for link in data['links']:
                print(f"  • {link}")
        
        # Print images if available
        if 'image' in data:
            print("\n🖼️ IMAGES:")
            print("-" * 80)
            print(f"  • {data['image']}")
        
        # Print raw data if available
        if 'raw_text' in data:
            print("\n📄 RAW TEXT:")
            print("-" * 80)
            print(data['raw_text'])
        
        # Print full JSON for debugging
        print("\n🔍 FULL JSON DATA:")
        print("-" * 80)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
    else:
        print("❌ Failed to extract content from the page")
        # Try basic extraction without JSON
        print("\nTrying basic extraction...")
        basic_result = extract(downloaded)
        if basic_result:
            print(basic_result)
        else:
            print("❌ Basic extraction also failed")
else:
    print("❌ Failed to fetch the URL")