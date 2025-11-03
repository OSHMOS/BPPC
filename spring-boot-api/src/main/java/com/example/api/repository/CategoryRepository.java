package com.example.api.repository;

import com.example.api.model.Category;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CategoryRepository extends JpaRepository<Category, Long> {
    Optional<Category> findByName(String name);
    Optional<Category> findBySlug(String slug);
    List<Category> findByParentCategoryId(Long parentCategoryId);
    List<Category> findByActive(Boolean active);
}
