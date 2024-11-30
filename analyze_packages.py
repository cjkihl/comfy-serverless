import importlib.metadata
from collections import defaultdict

def find_duplicate_versions():
    # Create a dictionary to store package versions
    package_versions = defaultdict(set)

    # Iterate over installed packages
    for dist in importlib.metadata.distributions():
        package_versions[dist.metadata['Name']].add(dist.version)

    # Print the number of packages
    print(f"Number of packages: {len(package_versions)}")

    # Print all packages and their versions
    for package, versions in package_versions.items():
        print(f"Package '{package}' has versions: {', '.join(versions)}")
        
    # Find and print packages with multiple versions
    for package, versions in package_versions.items():
        if len(versions) > 1:
            print(f"Package '{package}' has multiple versions: {', '.join(versions)}")

if __name__ == "__main__":
    find_duplicate_versions()