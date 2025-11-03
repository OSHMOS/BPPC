package com.example.api.repository;

import com.example.api.model.Post;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PostRepository extends JpaRepository<Post, Long> {
    List<Post> findByAuthorId(Long authorId);
    List<Post> findByCategory(String category);
    List<Post> findByPublished(Boolean published);
    List<Post> findByTitleContaining(String title);
}
