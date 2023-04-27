#if !defined(LIBRARIES_MAP_FUNCTIONS_HPP_)
#define LIBRARIES_MAP_FUNCTIONS_HPP_

#include <map>

/**
 * @brief If there already is a value associated with the key, return it; otherwise associate the given value with the key and return it.
 *
 * @tparam K Key type
 * @tparam V Value type
 * @param map Map
 * @param out_value Value associated with the key
 * @param key Key
 * @param value Value
 * @return true if a value was added
 * @return false if a value was not added
 */
template<typename K, typename V>
bool get_or_add(std::map<K, V>& map, V& out_value, const K& key, V& value)
{
    const auto map_handler = map.find(key);

    if (map_handler != map.end())
    {
        out_value = map_handler->second;
        return false;
    }
    else
    {
        map.insert(std::make_pair(key, value));
        out_value = value;
        return true;
    }
}

/**
 * @brief If there already is a value associated with the key, return it; otherwise associate the given value with the key and return it.
 *
 * @tparam K Key type
 * @tparam V Value type
 * @param map Map
 * @param key Key
 * @param value Value
 * @return V& Value associated with the key
 */
template<typename K, typename V>
const V& get_or_add(std::map<K, V>& map, const K& key, const V& value)
{
    const auto map_handler = map.find(key);

    if (map_handler != map.end())
    {
        return map_handler->second;
    }
    else
    {
        map.insert(std::make_pair(key, value));
        return value;
    }
}

/**
 * @brief If there already is a value associated with the key, return it; otherwise associate the default value with the key and return it.
 *
 * @tparam K Key type
 * @tparam V Value type
 * @param map Map
 * @param key Key
 * @return V& Value associated with the key
 */
template<typename K, typename V>
V& get_or_add(std::map<K, V>& map, const K& key)
{
    const auto map_handler = map.find(key);

    if (map_handler != map.end())
    {
        return map_handler->second;
    }
    else
    {
        map.insert(std::make_pair(key, V()));
        return map[key];
    }
}

#endif  // LIBRARIES_MAP_FUNCTIONS_HPP_
